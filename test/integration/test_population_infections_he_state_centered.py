"""
Integration test: PopulationInfections H+E model with state-centered AR(1) Rt.

Mirrors ``test_population_infections_he.py`` but with the inner temporal
process configured as ``AR1(parameterization='state')``. Same synthetic
126-day CA data, same priors, same observation models, same MCMC settings.
Verifies that the state-centered path produces statistically equivalent
posterior recovery to the innovation-form path.
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


class TestModelFit:
    """Fit the state-centered H+E model and check posterior recovery."""

    @pytest.fixture(scope="class")
    def fitted_model(
        self,
        he_model_state_centered: MultiSignalModel,
        daily_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> MultiSignalModel:
        """
        Fit the state-centered model to synthetic data via MCMC.

        Parameters
        ----------
        he_model_state_centered : MultiSignalModel
            State-centered H+E model fixture from ``conftest.py``.
        daily_hosp : pl.DataFrame
            Daily hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.

        Returns
        -------
        MultiSignalModel
            Model with MCMC results attached.
        """
        hosp_obs = he_model_state_centered.pad_observations(
            jnp.array(daily_hosp["daily_hosp_admits"].to_numpy(), dtype=jnp.float32)
        )
        ed_obs = he_model_state_centered.pad_observations(
            jnp.array(daily_ed["ed_visits"].to_numpy(), dtype=jnp.float32)
        )

        population_size = float(daily_hosp["pop"][0])

        he_model_state_centered.run(
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

        samples = he_model_state_centered.mcmc.get_samples()
        jax.block_until_ready(samples)
        return he_model_state_centered

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
            ArviZ DataTree with posterior group, initialization period trimmed.
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

    def test_state_site_present_innovation_sites_absent(
        self,
        fitted_model: MultiSignalModel,
    ) -> None:
        """
        Confirm the fit used the state-centered path.

        Parameters
        ----------
        fitted_model : MultiSignalModel
            Model with MCMC results.
        """
        samples = fitted_model.mcmc.get_samples()
        state_sites = [k for k in samples if k.endswith("_state")]
        noise_sites = [k for k in samples if k.endswith("_noise")]
        assert state_sites, f"Expected a _state site; got {sorted(samples.keys())}"
        assert not noise_sites, (
            f"Expected no _noise sites under state mode; got {noise_sites}"
        )

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
