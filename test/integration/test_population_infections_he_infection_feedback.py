"""
Integration test: PopulationInfections H+E model with infection feedback.

Checks that a builder-configured infection process is used inside the full
PopulationInfections + hospital + ED model graph.
"""

from __future__ import annotations

from datetime import date

import jax.numpy as jnp
import numpyro
import polars as pl
import pytest

from pyrenew.latent import InfectionsWithFeedback
from pyrenew.model import MultiSignalModel

pytestmark = pytest.mark.integration


N_DAYS_FIT = 126
OBS_START_DATE = date(2023, 11, 6)


class TestPriorPredictiveStructure:
    """Verify infection feedback is wired through the H+E model graph."""

    def test_infection_feedback_effective_rt_site(
        self,
        he_model_with_infection_feedback: MultiSignalModel,
        daily_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> None:
        """
        Single prior-predictive execution records raw and feedback-adjusted Rt.

        Parameters
        ----------
        he_model_with_infection_feedback : MultiSignalModel
            Built H+E model configured with ``InfectionsWithFeedback``.
        daily_hosp : pl.DataFrame
            Daily hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.
        """
        model = he_model_with_infection_feedback
        hosp_obs = model.pad_observations(
            jnp.array(daily_hosp["daily_hosp_admits"].to_numpy(), dtype=jnp.float32)
        )
        ed_obs = model.pad_observations(
            jnp.array(daily_ed["ed_visits"].to_numpy(), dtype=jnp.float32)
        )
        population_size = float(daily_hosp["pop"][0])

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as trace:
                model.sample(
                    n_days_post_init=N_DAYS_FIT,
                    population_size=population_size,
                    obs_start_date=OBS_START_DATE,
                    hospital={"obs": hosp_obs},
                    ed={"obs": ed_obs},
                )

        n_init = model.latent.n_initialization_points
        n_total = n_init + N_DAYS_FIT
        rt_raw = trace["PopulationInfections::rt_single"]["value"]
        rt_effective = trace["PopulationInfections::rt_single_effective"]["value"]

        assert isinstance(model.latent.infection_process, InfectionsWithFeedback)
        assert rt_raw.shape == (n_total, 1)
        assert rt_effective.shape == (n_total, 1)
        assert jnp.allclose(rt_effective[:n_init], rt_raw[:n_init])
        assert jnp.max(jnp.abs(rt_effective[n_init:] - rt_raw[n_init:])) > 1e-8
        assert trace["latent_infections"]["value"].shape == (n_total,)
        assert trace["hospital_predicted"]["value"].shape == (n_total,)
        assert trace["ed_predicted"]["value"].shape == (n_total,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
