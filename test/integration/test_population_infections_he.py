"""
Integration test: PopulationInfections H+E model with posterior recovery.

Fits a PopulationInfections model with hospital admissions and ED visit
observation processes to synthetic 126-day CA data, then checks that
posterior estimates recover known true parameters.
"""

from __future__ import annotations

from datetime import date

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


class TestDataAssembly:
    """Verify synthetic data can be loaded and aligned to the model."""

    def test_data_shapes(
        self,
        daily_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
        daily_infections: pl.DataFrame,
    ) -> None:
        """
        Verify synthetic data files have expected row counts and columns.

        Parameters
        ----------
        daily_hosp : pl.DataFrame
            Daily hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.
        daily_infections : pl.DataFrame
            True infections and R(t).
        """
        assert len(daily_hosp) == 126
        assert len(daily_ed) == 126
        assert len(daily_infections) == 126
        assert "daily_hosp_admits" in daily_hosp.columns
        assert "ed_visits" in daily_ed.columns
        assert "true_rt" in daily_infections.columns

    def test_observation_padding(
        self,
        he_model: MultiSignalModel,
        daily_hosp: pl.DataFrame,
    ) -> None:
        """
        Verify pad_observations produces correct shape and NaN prefix.

        Parameters
        ----------
        he_model : MultiSignalModel
            Built model.
        daily_hosp : pl.DataFrame
            Daily hospital admissions.
        """
        hosp_array = jnp.array(
            daily_hosp["daily_hosp_admits"].to_numpy(), dtype=jnp.float32
        )
        padded = he_model.pad_observations(hosp_array)
        n_init = he_model.latent.n_initialization_points

        assert padded.shape[0] == n_init + N_DAYS_FIT
        assert jnp.all(jnp.isnan(padded[:n_init]))
        assert not jnp.any(jnp.isnan(padded[n_init:]))

    def test_ed_model_includes_day_of_week_effect(
        self,
        he_model: MultiSignalModel,
    ) -> None:
        """
        Verify the ED observation model matches the synthetic data generator.

        Parameters
        ----------
        he_model : MultiSignalModel
            Built model.
        """
        assert he_model.observations["ed"].day_of_week_rv is not None


class TestModelFit:
    """Fit the H+E model and check posterior recovery."""

    @pytest.fixture(scope="class")
    def fitted_model(
        self,
        he_model: MultiSignalModel,
        daily_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> MultiSignalModel:
        """
        Fit the model to synthetic data via MCMC.

        Parameters
        ----------
        he_model : MultiSignalModel
            Built model.
        daily_hosp : pl.DataFrame
            Daily hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.

        Returns
        -------
        MultiSignalModel
            Model with MCMC results attached.
        """
        hosp_obs = he_model.pad_observations(
            jnp.array(daily_hosp["daily_hosp_admits"].to_numpy(), dtype=jnp.float32)
        )
        ed_obs = he_model.pad_observations(
            jnp.array(daily_ed["ed_visits"].to_numpy(), dtype=jnp.float32)
        )

        population_size = float(daily_hosp["pop"][0])

        he_model.run(
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,
            rng_key=random.PRNGKey(42),
            mcmc_args={"num_chains": NUM_CHAINS, "progress_bar": False},
            n_days_post_init=N_DAYS_FIT,
            population_size=population_size,
            obs_start_date=date(2023, 11, 6),
            hospital={"obs": hosp_obs},
            ed={"obs": ed_obs},
        )

        samples = he_model.mcmc.get_samples()
        jax.block_until_ready(samples)
        return he_model

    @pytest.fixture(scope="class")
    def posterior_dt(
        self,
        fitted_model: MultiSignalModel,
    ):
        """
        Convert MCMC samples to an ArviZ DataTree.

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
                "hospital_predicted": ["time"],
                "ed_predicted": ["time"],
            },
        )

        def trim_init(ds):
            """
            Trim initialization period from time dimension.

            Parameters
            ----------
            ds : xarray.Dataset
                Dataset to trim.

            Returns
            -------
            xarray.Dataset
                Trimmed dataset.
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
        Check that all parameters have acceptable Rhat and ESS.

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
        Check that 90% credible interval for R(t) covers the true value
        for at least 80% of time points.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        daily_infections : pl.DataFrame
            True infections and R(t) trajectory.
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
        Check posterior infection trajectory has correct shape and is positive.

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

    def test_predicted_observations_reasonable(
        self,
        posterior_dt,
        daily_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> None:
        """
        Check that posterior predicted observations have the right
        shape and plausible magnitude relative to observed data.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        daily_hosp : pl.DataFrame
            Daily hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.
        """
        hosp_pred = posterior_dt.posterior["hospital_predicted"]
        ed_pred = posterior_dt.posterior["ed_predicted"]

        assert hosp_pred.sizes["time"] == N_DAYS_FIT
        assert ed_pred.sizes["time"] == N_DAYS_FIT

        hosp_pred_median = float(hosp_pred.median(dim=["chain", "draw", "time"]).values)
        ed_pred_median = float(ed_pred.median(dim=["chain", "draw", "time"]).values)
        hosp_obs_mean = float(daily_hosp["daily_hosp_admits"].mean())
        ed_obs_mean = float(daily_ed["ed_visits"].mean())

        assert hosp_pred_median > 0, "Hospital predictions should be positive"
        assert ed_pred_median > 0, "ED predictions should be positive"
        assert hosp_obs_mean / 10 <= hosp_pred_median <= hosp_obs_mean * 10
        assert ed_obs_mean / 10 <= ed_pred_median <= ed_obs_mean * 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
