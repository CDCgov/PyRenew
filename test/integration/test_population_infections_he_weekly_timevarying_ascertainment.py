"""
Integration test: weekly hospital + daily ED model with time-varying ascertainment.

This exercises the mixed-cadence H+E structure with a
``TimeVaryingAscertainment`` model shared by the hospital and ED visit
observation processes.
"""

from __future__ import annotations

from datetime import date

import jax.numpy as jnp
import numpyro
import polars as pl
import pytest

from pyrenew.ascertainment import AscertainmentSignal, TimeVaryingAscertainment
from pyrenew.model import MultiSignalModel
from pyrenew.time import MMWR_WEEK

pytestmark = pytest.mark.integration


N_DAYS_FIT = 126
# First observation day of the synthetic data. 2023-11-05 is a Sunday.
OBS_START_DATE = date(2023, 11, 5)


def _build_hospital_obs_on_period_grid(
    model: MultiSignalModel,
    weekly_values: jnp.ndarray,
    first_day_dow: int,
) -> jnp.ndarray:
    """
    Build a dense weekly-observation array on the model's period grid.

    Parameters
    ----------
    model : MultiSignalModel
        Built model exposing ``latent.n_initialization_points``.
    weekly_values : jnp.ndarray
        Observed weekly hospital admissions, one per MMWR epiweek.
    first_day_dow : int
        Day-of-week index of element 0 of the shared daily axis.

    Returns
    -------
    jnp.ndarray
        Dense array with NaN for unobserved pre-data periods.
    """
    hosp = model.observations["hospital"]
    n_init = model.latent.n_initialization_points
    n_total = n_init + N_DAYS_FIT
    offset = hosp._compute_period_offset(first_day_dow, hosp.start_dow)
    n_periods = (n_total - offset) // hosp.aggregation_period
    n_pre = n_periods - len(weekly_values)
    return jnp.concatenate([jnp.full(n_pre, jnp.nan, dtype=jnp.float32), weekly_values])


class TestTimeVaryingStructure:
    """Check that the fixture has the intended H+E time-varying structure."""

    def test_observation_cadences(
        self,
        he_weekly_timevarying_ascertainment_model: MultiSignalModel,
    ) -> None:
        """
        Verify hospital is weekly MMWR and ED visits remain daily.

        Parameters
        ----------
        he_weekly_timevarying_ascertainment_model : MultiSignalModel
            Built model with time-varying ascertainment.
        """
        model = he_weekly_timevarying_ascertainment_model

        hospital = model.observations["hospital"]
        assert hospital.aggregation == "weekly"
        assert hospital.reporting_schedule == "regular"
        assert hospital.start_dow == MMWR_WEEK

        ed_visits = model.observations["ed_visits"]
        assert ed_visits.aggregation == "daily"
        assert ed_visits.day_of_week_rv is not None

    def test_timevarying_ascertainment_is_registered(
        self,
        he_weekly_timevarying_ascertainment_model: MultiSignalModel,
    ) -> None:
        """
        Verify both count observations use accessors from the same model.

        Parameters
        ----------
        he_weekly_timevarying_ascertainment_model : MultiSignalModel
            Built model with time-varying ascertainment.
        """
        model = he_weekly_timevarying_ascertainment_model
        assert set(model.ascertainment_models) == {"he_timevarying_ascertainment"}

        ascertainment = model.ascertainment_models["he_timevarying_ascertainment"]
        assert isinstance(ascertainment, TimeVaryingAscertainment)
        assert ascertainment.signals == ("hospital", "ed_visits")

        hospital_rate = model.observations["hospital"].ascertainment_rate_rv
        ed_rate = model.observations["ed_visits"].ascertainment_rate_rv
        assert isinstance(hospital_rate, AscertainmentSignal)
        assert isinstance(ed_rate, AscertainmentSignal)
        assert hospital_rate.ascertainment_name == "he_timevarying_ascertainment"
        assert hospital_rate.signal_name == "hospital"
        assert ed_rate.ascertainment_name == "he_timevarying_ascertainment"
        assert ed_rate.signal_name == "ed_visits"

    def test_weekly_obs_alignment(
        self,
        he_weekly_timevarying_ascertainment_model: MultiSignalModel,
        weekly_hosp: pl.DataFrame,
    ) -> None:
        """
        Verify weekly hospital observations align to the dense period grid.

        Parameters
        ----------
        he_weekly_timevarying_ascertainment_model : MultiSignalModel
            Built model with time-varying ascertainment.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        """
        model = he_weekly_timevarying_ascertainment_model
        first_day_dow = model._resolve_first_day_dow(OBS_START_DATE)
        weekly_values = jnp.array(
            weekly_hosp["weekly_hosp_admits"].to_numpy(), dtype=jnp.float32
        )
        hosp_obs = _build_hospital_obs_on_period_grid(
            model, weekly_values, first_day_dow
        )

        assert int((~jnp.isnan(hosp_obs)).sum()) == len(weekly_hosp)
        assert jnp.isnan(hosp_obs[0])
        assert not jnp.isnan(hosp_obs[-1])


class TestPriorPredictiveStructure:
    """Check the NumPyro graph for time-varying ascertainment and mixed cadence."""

    def test_timevarying_ascertainment_sites_are_sampled_once(
        self,
        he_weekly_timevarying_ascertainment_model: MultiSignalModel,
        weekly_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> None:
        """
        Single model execution exposes two weekly trajectories and two rates.

        Parameters
        ----------
        he_weekly_timevarying_ascertainment_model : MultiSignalModel
            Built model with time-varying ascertainment.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.
        """
        model = he_weekly_timevarying_ascertainment_model
        first_day_dow = model._resolve_first_day_dow(OBS_START_DATE)
        weekly_values = jnp.array(
            weekly_hosp["weekly_hosp_admits"].to_numpy(), dtype=jnp.float32
        )
        hosp_obs = _build_hospital_obs_on_period_grid(
            model, weekly_values, first_day_dow
        )
        ed_obs = model.pad_observations(
            jnp.array(daily_ed["ed_visits"].to_numpy(), dtype=jnp.float32)
        )
        population_size = float(weekly_hosp["pop"][0])

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as trace:
                model.sample(
                    n_days_post_init=N_DAYS_FIT,
                    population_size=population_size,
                    obs_start_date=OBS_START_DATE,
                    hospital={"obs": hosp_obs},
                    ed_visits={"obs": ed_obs},
                )

        n_total = model.latent.n_initialization_points + N_DAYS_FIT
        assert trace["he_timevarying_ascertainment_hospital_weekly"]["type"] == (
            "deterministic"
        )
        assert trace["he_timevarying_ascertainment_ed_visits_weekly"]["type"] == (
            "deterministic"
        )
        assert trace["he_timevarying_ascertainment_hospital"]["type"] == (
            "deterministic"
        )
        assert trace["he_timevarying_ascertainment_ed_visits"]["type"] == (
            "deterministic"
        )
        assert trace["he_timevarying_ascertainment_hospital"]["value"].shape == (
            n_total,
        )
        assert trace["he_timevarying_ascertainment_ed_visits"]["value"].shape == (
            n_total,
        )
        assert (
            trace["he_timevarying_ascertainment_hospital_weekly"]["value"].shape[-1]
            == 1
        )
        assert (
            trace["he_timevarying_ascertainment_ed_visits_weekly"]["value"].shape[-1]
            == 1
        )
        assert trace["hospital_predicted_daily"]["value"].shape == (n_total,)
        assert trace["ed_visits_predicted"]["value"].shape == (n_total,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
