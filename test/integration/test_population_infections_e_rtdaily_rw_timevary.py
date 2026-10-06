"""Integration test for daily Rt and time-varying ED ascertainment."""

from __future__ import annotations

from datetime import date

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import polars as pl
import pytest

from pyrenew.ascertainment import AscertainmentSignal, IndependentAscertainment
from pyrenew.latent import WeeklyTemporalProcess
from pyrenew.model import MultiSignalModel

pytestmark = pytest.mark.integration


N_DAYS_FIT = 126
N_DAYS_MCMC_SMOKE = 28
NUM_WARMUP = 2
NUM_SAMPLES = 2
OBS_START_DATE = date(2023, 11, 5)


def _padded_ed_observations(
    model: MultiSignalModel,
    daily_ed: pl.DataFrame,
    n_days: int,
) -> jax.Array:
    """Return synthetic ED observations on the model's padded time axis.

    Parameters
    ----------
    model
        Model providing the required initialization padding.
    daily_ed
        Synthetic daily ED visit data.
    n_days
        Number of observation days to include.

    Returns
    -------
    jax.Array
        Padded ED observation vector.
    """
    values = jnp.array(daily_ed["ed_visits"][:n_days].to_numpy(), dtype=jnp.float32)
    return model.pad_observations(values)


class TestModelStructure:
    """Verify the production-like ascertainment structure."""

    def test_time_varying_independent_ascertainment_is_registered(
        self,
        e_rtdaily_rw_timevary_model: MultiSignalModel,
    ) -> None:
        """Verify the ED observation uses the registered ascertainment signal."""
        model = e_rtdaily_rw_timevary_model

        assert set(model.ascertainment_models) == {"ed_ascertainment"}
        ascertainment = model.ascertainment_models["ed_ascertainment"]
        assert isinstance(ascertainment, IndependentAscertainment)
        assert ascertainment.signals == ("ed",)
        assert isinstance(ascertainment.temporal_processes["ed"], WeeklyTemporalProcess)
        assert ascertainment.requires_calendar_anchor()

        rate = model.observations["ed"].ascertainment_rate_rv
        assert isinstance(rate, AscertainmentSignal)
        assert rate.ascertainment_name == "ed_ascertainment"
        assert rate.signal_name == "ed"


class TestSyntheticExecution:
    """Run the model against the existing constant-ascertainment data."""

    def test_full_graph_records_time_varying_ascertainment(
        self,
        e_rtdaily_rw_timevary_model: MultiSignalModel,
        daily_ed: pl.DataFrame,
        true_params: dict,
    ) -> None:
        """Verify one conditioned execution records valid ascertainment sites."""
        model = e_rtdaily_rw_timevary_model
        ed_obs = _padded_ed_observations(model, daily_ed, N_DAYS_FIT)
        population_size = float(true_params["population"])

        model.validate_data(
            n_days_post_init=N_DAYS_FIT,
            obs_start_date=OBS_START_DATE,
            ed={"obs": ed_obs},
        )
        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as trace:
                model.sample(
                    n_days_post_init=N_DAYS_FIT,
                    population_size=population_size,
                    obs_start_date=OBS_START_DATE,
                    ed={"obs": ed_obs},
                )

        n_total = model.latent.n_initialization_points + N_DAYS_FIT
        baseline = trace["ed_ascertainment_baseline_ed"]["value"]
        weekly = trace["ed_ascertainment_ed_weekly"]["value"]
        trajectory = trace["ed_ascertainment_ed"]["value"]
        predicted = trace["ed_predicted"]["value"]

        assert trace["p_ed_visit_mean"]["type"] == "sample"
        assert baseline.shape == ()
        assert weekly.ndim == 2
        assert weekly.shape[1] == 1
        assert weekly.shape[0] < n_total
        assert trajectory.shape == (n_total,)
        assert jnp.isfinite(baseline)
        assert 0 < baseline < 1
        assert jnp.all(jnp.isfinite(trajectory))
        assert jnp.all((trajectory > 0) & (trajectory < 1))
        assert predicted.shape == (n_total,)
        assert jnp.all(jnp.isfinite(predicted[model.latent.n_initialization_points :]))
        assert trace["ed_obs"]["is_observed"]

    def test_short_mcmc_fit_runs(
        self,
        e_rtdaily_rw_timevary_model: MultiSignalModel,
        daily_ed: pl.DataFrame,
        true_params: dict,
    ) -> None:
        """Verify NUTS can initialize and sample the integrated model."""
        model = e_rtdaily_rw_timevary_model
        ed_obs = _padded_ed_observations(model, daily_ed, N_DAYS_MCMC_SMOKE)

        model.run(
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,
            rng_key=random.PRNGKey(42),
            mcmc_args={"progress_bar": False},
            n_days_post_init=N_DAYS_MCMC_SMOKE,
            population_size=float(true_params["population"]),
            obs_start_date=OBS_START_DATE,
            ed={"obs": ed_obs},
        )

        assert model.mcmc is not None
        samples = model.mcmc.get_samples()
        jax.block_until_ready(samples)

        required_sites = {
            "p_ed_visit_mean",
            "autoreg_p_ed_visit",
            "p_ed_visit_w_sd",
            "ed_ascertainment_ed",
            "ed_predicted",
        }
        assert required_sites <= set(samples)
        for site in required_sites - {"ed_predicted"}:
            assert jnp.all(jnp.isfinite(samples[site]))
        assert jnp.all(
            jnp.isfinite(
                samples["ed_predicted"][:, model.latent.n_initialization_points :]
            )
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
