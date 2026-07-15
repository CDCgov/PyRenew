"""
Integration tests for weekly hospital and daily ED ratio-linked ascertainment.
"""

from __future__ import annotations

from datetime import date

import jax.numpy as jnp
import numpyro
import polars as pl
import pytest

from pyrenew.ascertainment import AscertainmentSignal, RatioLinkedAscertainment
from pyrenew.model import MultiSignalModel
from pyrenew.time import MMWR_WEEK

pytestmark = pytest.mark.integration


N_DAYS_FIT = 126
OBS_START_DATE = date(2023, 11, 5)


def _build_hospital_obs_on_period_grid(
    model: MultiSignalModel,
    weekly_values: jnp.ndarray,
    first_day_dow: int,
) -> jnp.ndarray:
    """
    Build a dense hospital observation array on the model's period grid.

    Parameters
    ----------
    model : MultiSignalModel
        Built model exposing its initialization period.
    weekly_values : jnp.ndarray
        Observed hospital admissions by MMWR epiweek.
    first_day_dow : int
        Day-of-week index of the first element on the shared daily axis.

    Returns
    -------
    jnp.ndarray
        Weekly observation array padded with missing pre-data periods.
    """
    hospital = model.observations["hospital"]
    n_total = model.latent.n_initialization_points + N_DAYS_FIT
    offset = hospital._compute_period_offset(first_day_dow, hospital.start_dow)
    n_periods = (n_total - offset) // hospital.aggregation_period
    n_pre = n_periods - len(weekly_values)
    return jnp.concatenate([jnp.full(n_pre, jnp.nan, dtype=jnp.float32), weekly_values])


class TestRatioLinkedStructure:
    """Check the configured ratio-linked ascertainment structure."""

    def test_ratio_linked_ascertainment_is_registered(
        self,
        he_weekly_ratio_linked_ascertainment_model: MultiSignalModel,
    ) -> None:
        """
        Verify the observations use accessors from the ratio-linked model.

        Parameters
        ----------
        he_weekly_ratio_linked_ascertainment_model : MultiSignalModel
            Built model with ratio-linked ascertainment.
        """
        model = he_weekly_ratio_linked_ascertainment_model
        assert set(model.ascertainment_models) == {"he_ascertainment"}

        ascertainment = model.ascertainment_models["he_ascertainment"]
        assert isinstance(ascertainment, RatioLinkedAscertainment)
        assert ascertainment.signals == ("ed_visits", "hospital")
        assert ascertainment.base_signal == "ed_visits"
        assert ascertainment.linked_signal == "hospital"

        hospital_rate = model.observations["hospital"].ascertainment_rate_rv
        ed_rate = model.observations["ed_visits"].ascertainment_rate_rv
        assert isinstance(hospital_rate, AscertainmentSignal)
        assert isinstance(ed_rate, AscertainmentSignal)
        assert hospital_rate.ascertainment_name == "he_ascertainment"
        assert hospital_rate.signal_name == "hospital"
        assert ed_rate.ascertainment_name == "he_ascertainment"
        assert ed_rate.signal_name == "ed_visits"

        hospital = model.observations["hospital"]
        assert hospital.aggregation == "weekly"
        assert hospital.reporting_schedule == "regular"
        assert hospital.start_dow == MMWR_WEEK


class TestRatioLinkedModelExecution:
    """Check ratio-linked sites during a complete model execution."""

    def test_ratio_linked_rates_and_prediction_shapes(
        self,
        he_weekly_ratio_linked_ascertainment_model: MultiSignalModel,
        weekly_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> None:
        """
        Verify sampled rates and mixed-cadence observation predictions.

        Parameters
        ----------
        he_weekly_ratio_linked_ascertainment_model : MultiSignalModel
            Built model with ratio-linked ascertainment.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.
        """
        model = he_weekly_ratio_linked_ascertainment_model
        first_day_dow = model._resolve_first_day_dow(OBS_START_DATE)
        weekly_values = jnp.array(
            weekly_hosp["weekly_hosp_admits"].to_numpy(), dtype=jnp.float32
        )
        hosp_obs = _build_hospital_obs_on_period_grid(
            model,
            weekly_values,
            first_day_dow,
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

        base_rate = trace["iedr"]["value"]
        ratio = trace["ihr_rel_iedr"]["value"]
        ed_rate = trace["he_ascertainment_ed_visits"]["value"]
        hospital_rate = trace["he_ascertainment_hospital"]["value"]

        assert trace["iedr"]["type"] == "sample"
        assert trace["ihr_rel_iedr"]["type"] == "sample"
        assert trace["he_ascertainment_ed_visits"]["type"] == "deterministic"
        assert trace["he_ascertainment_hospital"]["type"] == "deterministic"
        assert jnp.allclose(ed_rate, base_rate)
        assert jnp.allclose(hospital_rate, base_rate * ratio)

        n_total = model.latent.n_initialization_points + N_DAYS_FIT
        hospital = model.observations["hospital"]
        offset = hospital._compute_period_offset(first_day_dow, hospital.start_dow)
        n_periods = (n_total - offset) // hospital.aggregation_period
        assert trace["hospital_predicted"]["value"].shape == (n_periods,)
        assert trace["hospital_predicted_daily"]["value"].shape == (n_total,)
        assert trace["ed_visits_predicted"]["value"].shape == (n_total,)
