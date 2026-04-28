"""
Integration test: PopulationInfections H+E model with WEEKLY R(t).

Mirrors ``test_population_infections_he_weekly`` but parameterizes R(t)
weekly via ``WeeklyTemporalProcess`` and broadcasts to the daily renewal axis.
This is the production pyrenew-hew configuration: weekly hospital admissions +
daily ED visits + weekly calendar-aligned R(t).
"""

from __future__ import annotations

from datetime import date

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import polars as pl
import pytest

from pyrenew.model import MultiSignalModel

pytestmark = pytest.mark.integration


N_DAYS_FIT = 126
NUM_WARMUP = 500
NUM_SAMPLES = 500
NUM_CHAINS = 4
# First observation day of the synthetic data. 2023-11-05 is a Sunday (ISO dow = 6).
OBS_START_DATE = date(2023, 11, 5)
WEEK_START_DOW = 6


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


def _expected_n_weekly(model: MultiSignalModel, first_day_dow: int) -> int:
    """
    Expected number of weekly R(t) samples for calendar-week alignment.

    Parameters
    ----------
    model : MultiSignalModel
        Built model exposing ``latent.n_initialization_points``.
    first_day_dow : int
        Day-of-week index of element 0 of the shared daily axis.

    Returns
    -------
    int
        Number of weekly Rt samples covering the daily model axis.
    """
    n_total = model.latent.n_initialization_points + N_DAYS_FIT
    trim = (first_day_dow - WEEK_START_DOW) % 7
    return (n_total + trim + 6) // 7


class TestPriorPredictiveStructure:
    """Verify the weekly-Rt graph records a weekly trajectory at the right shape."""

    def test_weekly_rt_recorded(
        self,
        he_weekly_rt_model: MultiSignalModel,
        weekly_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> None:
        """
        Single prior-predictive sample exposes ``log_rt_single_weekly`` at the
        expected weekly length and a daily-length broadcast Rt.

        Parameters
        ----------
        he_weekly_rt_model : MultiSignalModel
            Built model with calendar-aligned weekly Rt.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.
        """
        first_day_dow = he_weekly_rt_model._resolve_first_day_dow(OBS_START_DATE)
        weekly_values = jnp.array(
            weekly_hosp["weekly_hosp_admits"].to_numpy(), dtype=jnp.float32
        )
        hosp_obs = _build_hospital_obs_on_period_grid(
            he_weekly_rt_model, weekly_values, first_day_dow
        )
        ed_obs = he_weekly_rt_model.pad_observations(
            jnp.array(daily_ed["ed_visits"].to_numpy(), dtype=jnp.float32)
        )
        population_size = float(weekly_hosp["pop"][0])

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as trace:
                he_weekly_rt_model.sample(
                    n_days_post_init=N_DAYS_FIT,
                    population_size=population_size,
                    obs_start_date=OBS_START_DATE,
                    hospital={"obs": hosp_obs},
                    ed={"obs": ed_obs},
                )

        n_total = he_weekly_rt_model.latent.n_initialization_points + N_DAYS_FIT
        n_weekly = _expected_n_weekly(he_weekly_rt_model, first_day_dow)

        weekly = trace["log_rt_single_weekly"]["value"]
        daily = trace["PopulationInfections::log_rt_single"]["value"]

        assert weekly.shape == (n_weekly, 1)
        assert daily.shape == (n_total, 1)
        assert n_weekly < n_total

        # Each block of 7 daily values past the leading partial week should be
        # constant (the calendar-week broadcast invariant).
        partial_len = (WEEK_START_DOW - first_day_dow) % 7
        if partial_len > 0:
            assert jnp.allclose(daily[:partial_len], daily[0])
        first_full = partial_len
        assert jnp.allclose(daily[first_full : first_full + 7], daily[first_full])


class TestModelFit:
    """Fit the weekly-Rt H+E model and check posterior recovery."""

    @pytest.fixture(scope="class")
    def fitted_model(
        self,
        he_weekly_rt_model: MultiSignalModel,
        weekly_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> MultiSignalModel:
        """
        Fit the weekly-Rt H+E model to synthetic data via MCMC.

        Parameters
        ----------
        he_weekly_rt_model : MultiSignalModel
            Built model with calendar-aligned weekly Rt.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.

        Returns
        -------
        MultiSignalModel
            Model with MCMC results attached.
        """
        first_day_dow = he_weekly_rt_model._resolve_first_day_dow(OBS_START_DATE)

        weekly_values = jnp.array(
            weekly_hosp["weekly_hosp_admits"].to_numpy(), dtype=jnp.float32
        )
        hosp_obs = _build_hospital_obs_on_period_grid(
            he_weekly_rt_model, weekly_values, first_day_dow
        )

        ed_obs = he_weekly_rt_model.pad_observations(
            jnp.array(daily_ed["ed_visits"].to_numpy(), dtype=jnp.float32)
        )

        population_size = float(weekly_hosp["pop"][0])

        he_weekly_rt_model.run(
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,
            rng_key=random.PRNGKey(42),
            mcmc_args={"num_chains": NUM_CHAINS, "progress_bar": False},
            n_days_post_init=N_DAYS_FIT,
            population_size=population_size,
            obs_start_date=OBS_START_DATE,
            hospital={"obs": hosp_obs},
            ed={"obs": ed_obs},
        )

        samples = he_weekly_rt_model.mcmc.get_samples()
        jax.block_until_ready(samples)
        return he_weekly_rt_model

    @pytest.fixture(scope="class")
    def posterior_dt(
        self,
        fitted_model: MultiSignalModel,
    ):
        """
        Convert MCMC samples to an ArviZ DataTree, trimming the init period.

        Parameters
        ----------
        fitted_model : MultiSignalModel
            Model with MCMC results.

        Returns
        -------
        xarray.DataTree
            ArviZ DataTree with posterior group.
        """
        n_init = fitted_model.latent.n_initialization_points
        dt = az.from_numpyro(
            fitted_model.mcmc,
            dims={
                "latent_infections": ["time"],
                "PopulationInfections::infections_aggregate": ["time"],
                "PopulationInfections::log_rt_single": ["time", "dummy"],
                "PopulationInfections::rt_single": ["time", "dummy"],
                "log_rt_single_weekly": ["rt_week", "dummy"],
                "hospital_predicted_daily": ["time"],
                "hospital_predicted": ["week"],
                "ed_predicted": ["time"],
            },
        )

        def trim_init(ds):
            """
            Trim the initialization period from the ``time`` dimension only.

            Parameters
            ----------
            ds
                Dataset to trim.

            Returns
            -------
            xarray.Dataset
                Dataset with ``time`` sliced to ``[n_init:]``; other dims
                pass through unchanged.
            """
            if "time" in ds.dims:
                ds = ds.isel(time=slice(n_init, None))
                ds = ds.assign_coords(time=range(ds.sizes["time"]))
            return ds

        return dt.map_over_datasets(trim_init)

    def test_mcmc_convergence(
        self,
        posterior_dt,
    ) -> None:
        """
        Check that core parameters have acceptable Rhat and ESS.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        """
        summary = az.summary(
            posterior_dt,
            var_names=["I0", "log_rt_time_0", "ihr", "iedr"],
        )
        rhat = summary["r_hat"].astype(float)
        ess = summary["ess_bulk"].astype(float)
        assert (rhat < 1.05).all(), f"Rhat exceeded 1.05:\n{summary[rhat >= 1.05]}"
        assert (ess > 100).all(), f"ESS_bulk below 100:\n{summary[ess <= 100]}"

    def test_weekly_rt_posterior_shape(
        self,
        fitted_model: MultiSignalModel,
        posterior_dt,
    ) -> None:
        """
        Check the weekly Rt site lives on the weekly cadence in the posterior.

        Parameters
        ----------
        fitted_model : MultiSignalModel
            Fitted model.
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        """
        first_day_dow = fitted_model._resolve_first_day_dow(OBS_START_DATE)
        n_weekly = _expected_n_weekly(fitted_model, first_day_dow)

        weekly = posterior_dt.posterior["log_rt_single_weekly"]
        assert weekly.sizes["rt_week"] == n_weekly

    def test_rt_posterior_covers_truth(
        self,
        posterior_dt,
        daily_infections: pl.DataFrame,
    ) -> None:
        """
        Check that the 90% credible interval for R(t) covers the true value
        for at least 80% of time points.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        daily_infections : pl.DataFrame
            True R(t) trajectory.
        """
        rt_posterior = posterior_dt.posterior["PopulationInfections::rt_single"]
        rt_q05 = rt_posterior.quantile(0.05, dim=["chain", "draw"]).values
        rt_q95 = rt_posterior.quantile(0.95, dim=["chain", "draw"]).values

        true_rt = daily_infections["true_rt"].to_numpy()

        if rt_q05.ndim > 1:
            rt_q05 = rt_q05.squeeze()
            rt_q95 = rt_q95.squeeze()

        n_compare = min(len(true_rt), len(rt_q05))
        covered = (true_rt[:n_compare] >= rt_q05[:n_compare]) & (
            true_rt[:n_compare] <= rt_q95[:n_compare]
        )
        coverage = float(np.mean(covered))
        assert coverage >= 0.80, (
            f"R(t) 90% CI coverage was {coverage:.1%}, expected >= 80%"
        )

    def test_ascertainment_rates_recover_order_of_magnitude(
        self,
        posterior_dt,
        true_params: dict,
    ) -> None:
        """
        Check that posterior median IHR and IEDR are within a factor
        of 5 of the true values.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        true_params : dict
            Ground-truth parameter dictionary.
        """
        true_ihr = true_params["hospitalizations"]["ihr"]
        true_iedr = true_params["ed_visits"]["iedr"]

        ihr_median = float(
            posterior_dt.posterior["ihr"].median(dim=["chain", "draw"]).values
        )
        iedr_median = float(
            posterior_dt.posterior["iedr"].median(dim=["chain", "draw"]).values
        )

        assert true_ihr / 5 <= ihr_median <= true_ihr * 5, (
            f"IHR median {ihr_median:.4f} not within 5x of true {true_ihr}"
        )
        assert true_iedr / 5 <= iedr_median <= true_iedr * 5, (
            f"IEDR median {iedr_median:.4f} not within 5x of true {true_iedr}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
