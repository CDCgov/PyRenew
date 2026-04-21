"""
Integration test: PopulationInfections H+E model with WEEKLY hospital admissions.

Same structure as ``test_population_infections_he`` but the hospital
signal is aggregated to MMWR epiweeks. Fits the mixed-cadence model
(weekly hospital + daily ED) to synthetic 126-day CA data and checks
posterior recovery.
"""

from __future__ import annotations

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import polars as pl
import pytest

from pyrenew.model import MultiSignalModel

pytestmark = pytest.mark.integration


N_DAYS_FIT = 126
NUM_WARMUP = 500
NUM_SAMPLES = 500
NUM_CHAINS = 4
# Day 0 of the synthetic data is 2023-11-05, a Sunday (ISO dow = 6).
OBS_START_DOW = 6


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
    offset = (hosp.period_end_dow + 1 - first_day_dow) % hosp.aggregation_period
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
        assert h.aggregation_period == 7
        assert h.reporting_schedule == "regular"
        assert h.period_end_dow == 5

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
        assert ed.aggregation_period == 1
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
        first_day_dow = he_weekly_model.compute_first_day_dow(OBS_START_DOW)
        weekly_values = jnp.array(
            weekly_hosp["weekly_hosp_admits"].to_numpy(), dtype=jnp.float32
        )
        hosp_obs = _build_hospital_obs_on_period_grid(
            he_weekly_model, weekly_values, first_day_dow
        )

        n_init = he_weekly_model.latent.n_initialization_points
        n_total = n_init + N_DAYS_FIT
        hosp = he_weekly_model.observations["hospital"]
        offset = (hosp.period_end_dow + 1 - first_day_dow) % hosp.aggregation_period
        n_periods = (n_total - offset) // hosp.aggregation_period

        assert hosp_obs.shape == (n_periods,)
        assert jnp.isnan(hosp_obs[0])
        assert not jnp.isnan(hosp_obs[-1])
        assert int((~jnp.isnan(hosp_obs)).sum()) == len(weekly_hosp)


class TestModelFit:
    """Fit the weekly H+E model and check posterior recovery."""

    @pytest.fixture(scope="class")
    def fitted_model(
        self,
        he_weekly_model: MultiSignalModel,
        weekly_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> MultiSignalModel:
        """
        Fit the mixed-cadence H+E model to synthetic data via MCMC.

        Parameters
        ----------
        he_weekly_model : MultiSignalModel
            Built model.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.

        Returns
        -------
        MultiSignalModel
            Model with MCMC results attached.
        """
        first_day_dow = he_weekly_model.compute_first_day_dow(OBS_START_DOW)

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

        he_weekly_model.run(
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,
            rng_key=random.PRNGKey(42),
            mcmc_args={"num_chains": NUM_CHAINS, "progress_bar": False},
            n_days_post_init=N_DAYS_FIT,
            population_size=population_size,
            hospital={"obs": hosp_obs, "first_day_dow": first_day_dow},
            ed={"obs": ed_obs, "first_day_dow": first_day_dow},
        )

        samples = he_weekly_model.mcmc.get_samples()
        jax.block_until_ready(samples)
        return he_weekly_model

    @pytest.fixture(scope="class")
    def posterior_dt(
        self,
        fitted_model: MultiSignalModel,
    ):
        """
        Convert MCMC samples to an ArviZ DataTree, trimming the init period.

        The hospital signal lives on the weekly period grid (dim ``week``);
        the daily-scale sites (``time``) are trimmed by ``n_init``.

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
                Dataset with ``time`` sliced to ``[n_init:]``; ``week``
                and other dims pass through unchanged.
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

    def test_infection_trajectory_shape(
        self,
        posterior_dt,
    ) -> None:
        """
        Check the posterior infection trajectory has correct shape and is positive.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        """
        infections = posterior_dt.posterior["latent_infections"]
        assert infections.sizes["time"] == N_DAYS_FIT
        assert (infections.values > 0).all()

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

    def test_hospital_predicted_weekly_grid(
        self,
        posterior_dt,
        weekly_hosp: pl.DataFrame,
    ) -> None:
        """
        Check that hospital posterior predictions live on the weekly grid
        and have plausible magnitude relative to observed weekly counts.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        """
        hosp_pred = posterior_dt.posterior["hospital_predicted"]
        # n_periods from the weekly grid must be >= the number of observed weeks.
        assert hosp_pred.sizes["week"] >= len(weekly_hosp)

        hosp_pred_median = float(hosp_pred.median(dim=["chain", "draw", "week"]).values)
        hosp_obs_mean = float(weekly_hosp["weekly_hosp_admits"].mean())

        assert hosp_pred_median > 0, "Hospital predictions should be positive"
        assert hosp_obs_mean / 10 <= hosp_pred_median <= hosp_obs_mean * 10

    def test_hospital_predicted_daily_has_daily_grid(
        self,
        posterior_dt,
    ) -> None:
        """
        Check that the ``hospital_predicted_daily`` deterministic site covers
        the daily shared axis.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        """
        hosp_pred_daily = posterior_dt.posterior["hospital_predicted_daily"]
        assert hosp_pred_daily.sizes["time"] == N_DAYS_FIT

    def test_ed_predicted_reasonable(
        self,
        posterior_dt,
        daily_ed: pl.DataFrame,
    ) -> None:
        """
        Check that daily ED posterior predictions have the right shape and
        plausible magnitude relative to observed daily ED visits.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        daily_ed : pl.DataFrame
            Daily ED visits.
        """
        ed_pred = posterior_dt.posterior["ed_predicted"]
        assert ed_pred.sizes["time"] == N_DAYS_FIT

        ed_pred_median = float(ed_pred.median(dim=["chain", "draw", "time"]).values)
        ed_obs_mean = float(daily_ed["ed_visits"].mean())

        assert ed_pred_median > 0, "ED predictions should be positive"
        assert ed_obs_mean / 10 <= ed_pred_median <= ed_obs_mean * 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
