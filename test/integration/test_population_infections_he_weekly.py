"""
Integration test: PopulationInfections H+E model with WEEKLY hospital admissions.

Same structure as ``test_population_infections_he`` but the hospital
signal is aggregated to MMWR epiweeks. Checks mixed-cadence structure
(weekly hospital + daily ED) on synthetic 126-day CA data without running
MCMC.
"""

from __future__ import annotations

from datetime import date

import jax.numpy as jnp
import numpyro
import polars as pl
import pytest

from pyrenew.model import MultiSignalModel
from pyrenew.time import MMWR_WEEK

pytestmark = pytest.mark.integration


N_DAYS_FIT = 126
# First observation day of the synthetic data. 2023-11-05 is a Sunday (ISO dow = 6).
OBS_START_DATE = date(2023, 11, 5)


def _build_hospital_obs_on_period_grid(
    model: MultiSignalModel,
    weekly_values: jnp.ndarray,
    first_day_dow: int,
) -> jnp.ndarray:
    """
    Build a dense weekly-observation array on the model's period grid.

    Pads ``n_pre`` NaN values at the front for periods that precede the
    first observed week (periods that overlap the initialization window
    and any pre-observation gap).

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
        Dense array of shape ``(n_periods,)`` with NaN for unobserved
        periods and observed counts for periods covered by
        ``weekly_values``.
    """
    hosp = model.observations["hospital"]
    n_init = model.latent.n_initialization_points
    n_total = n_init + N_DAYS_FIT
    offset = hosp._compute_period_offset(first_day_dow, hosp.start_dow)
    n_periods = (n_total - offset) // hosp.aggregation_period
    n_pre = n_periods - len(weekly_values)
    return jnp.concatenate([jnp.full(n_pre, jnp.nan, dtype=jnp.float32), weekly_values])


class TestDataAssembly:
    """Verify synthetic data can be loaded and aligned to the weekly model."""

    def test_data_shapes(
        self,
        weekly_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
        daily_infections: pl.DataFrame,
    ) -> None:
        """
        Verify synthetic data files have expected row counts and columns.

        Parameters
        ----------
        weekly_hosp : pl.DataFrame
            Weekly MMWR-epiweek hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.
        daily_infections : pl.DataFrame
            True infections and R(t).
        """
        assert len(weekly_hosp) == 18
        assert len(daily_ed) == 126
        assert len(daily_infections) == 126
        assert "weekly_hosp_admits" in weekly_hosp.columns
        assert "ed_visits" in daily_ed.columns
        assert "true_rt" in daily_infections.columns

    def test_hospital_is_weekly_regular(
        self,
        he_weekly_model: MultiSignalModel,
    ) -> None:
        """
        Verify the hospital observation is weekly-aggregated and MMWR-anchored.

        Parameters
        ----------
        he_weekly_model : MultiSignalModel
            Built model.
        """
        h = he_weekly_model.observations["hospital"]
        assert h.aggregation == "weekly"
        assert h.reporting_schedule == "regular"
        assert h.start_dow == MMWR_WEEK

    def test_ed_stays_daily(
        self,
        he_weekly_model: MultiSignalModel,
    ) -> None:
        """
        Verify the ED observation remains daily with a day-of-week effect.

        Parameters
        ----------
        he_weekly_model : MultiSignalModel
            Built model.
        """
        ed = he_weekly_model.observations["ed"]
        assert ed.aggregation == "daily"
        assert ed.day_of_week_rv is not None

    def test_weekly_obs_alignment(
        self,
        he_weekly_model: MultiSignalModel,
        weekly_hosp: pl.DataFrame,
    ) -> None:
        """
        Verify the weekly obs array has the correct dense-on-period-grid shape.

        Parameters
        ----------
        he_weekly_model : MultiSignalModel
            Built model.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        """
        first_day_dow = he_weekly_model._resolve_first_day_dow(OBS_START_DATE)
        weekly_values = jnp.array(
            weekly_hosp["weekly_hosp_admits"].to_numpy(), dtype=jnp.float32
        )
        hosp_obs = _build_hospital_obs_on_period_grid(
            he_weekly_model, weekly_values, first_day_dow
        )

        n_init = he_weekly_model.latent.n_initialization_points
        n_total = n_init + N_DAYS_FIT
        hosp = he_weekly_model.observations["hospital"]
        offset = hosp._compute_period_offset(first_day_dow, hosp.start_dow)
        n_periods = (n_total - offset) // hosp.aggregation_period

        assert hosp_obs.shape == (n_periods,)
        assert jnp.isnan(hosp_obs[0])
        assert not jnp.isnan(hosp_obs[-1])
        assert int((~jnp.isnan(hosp_obs)).sum()) == len(weekly_hosp)


class TestPriorPredictiveStructure:
    """Verify weekly-H+E graph structure without running MCMC."""

    def test_weekly_hospital_and_daily_ed_sites(
        self,
        he_weekly_model: MultiSignalModel,
        weekly_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> None:
        """
        Single prior-predictive sample exposes weekly hospital predictions,
        daily hospital predictions, and daily ED predictions on the expected grids.

        Parameters
        ----------
        he_weekly_model : MultiSignalModel
            Built model.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.
        """
        first_day_dow = he_weekly_model._resolve_first_day_dow(OBS_START_DATE)
        weekly_values = jnp.array(
            weekly_hosp["weekly_hosp_admits"].to_numpy(), dtype=jnp.float32
        )
        hosp_obs = _build_hospital_obs_on_period_grid(
            he_weekly_model, weekly_values, first_day_dow
        )
        ed_obs = he_weekly_model.pad_observations(
            jnp.array(daily_ed["ed_visits"].to_numpy(), dtype=jnp.float32)
        )
        population_size = float(weekly_hosp["pop"][0])

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as trace:
                he_weekly_model.sample(
                    n_days_post_init=N_DAYS_FIT,
                    population_size=population_size,
                    obs_start_date=OBS_START_DATE,
                    hospital={"obs": hosp_obs},
                    ed={"obs": ed_obs},
                )

        n_init = he_weekly_model.latent.n_initialization_points
        n_total = n_init + N_DAYS_FIT
        hosp = he_weekly_model.observations["hospital"]
        offset = hosp._compute_period_offset(first_day_dow, hosp.start_dow)
        n_periods = (n_total - offset) // hosp.aggregation_period

        assert trace["hospital_predicted"]["value"].shape == (n_periods,)
        assert trace["hospital_predicted_daily"]["value"].shape == (n_total,)
        assert trace["ed_predicted"]["value"].shape == (n_total,)
        assert n_periods >= len(weekly_hosp)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
