"""
Integration test: SubpopulationInfections H+W model with tutorial-data fitting and prediction.

Fits a SubpopulationInfections model with hospital admissions and
wastewater observation processes to California hospital data and
simulated NWSS wastewater data, then checks posterior diagnostics
and that predictions are consistent with the observed signals.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import arviz_base as az_base
import arviz_stats as az_stats
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import pytest
import xarray as xr
from numpyro.diagnostics import summary

from pyrenew.ascertainment import IndependentAscertainment
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import WeeklyTemporalProcess
from pyrenew.model import MultiSignalModel
from pyrenew.observation import PoissonNoise, SubpopulationCounts

pytestmark = pytest.mark.integration


START_DATE = date(2023, 1, 1)
N_DAYS_FIT: int = 90
N_DAYS_TAIL_TRIM: int = 14
NUM_WARMUP: int = 500
NUM_SAMPLES: int = 500
NUM_CHAINS: int = 4
SUBPOP_FRACTIONS: jnp.ndarray = jnp.array([0.10, 0.14, 0.21, 0.22, 0.07, 0.26])
WW_MONITORED_SUBPOPS: jnp.ndarray = jnp.array([0, 1, 2, 3, 4])

DataDict = dict[str, Any]
ObservationData = dict[str, DataDict]


def _prepare_observation_data(
    model: MultiSignalModel,
    n_days_fit: int,
    hosp_admits: jnp.ndarray,
    ww_data: DataDict,
    ww_monitored_subpops: jnp.ndarray,
) -> ObservationData:
    """
    Align hospital and wastewater observation data to the shared time axis.

    Parameters
    ----------
    model
        Built multi-signal model providing pad_observations and shift_times.
    n_days_fit
        Number of observation days to include in the fit.
    hosp_admits
        Dense daily hospital admissions array.
    ww_data
        Wastewater data dictionary from load_wastewater_data_for_state.
    ww_monitored_subpops
        Indices of subpopulations with wastewater monitoring.

    Returns
    -------
    dict
        Keyword-argument dictionary for model.run() with "hospital"
        and "wastewater" entries.
    """
    hosp_obs = model.pad_observations(
        jnp.array(hosp_admits[:n_days_fit], dtype=jnp.float32)
    )

    time_indices = jnp.array(ww_data["time_indices"])
    ww_mask = time_indices < n_days_fit
    ww_times = model.shift_times(time_indices[ww_mask])
    ww_conc = jnp.array(ww_data["observed_conc"])[ww_mask]
    ww_sensors = jnp.array(ww_data["site_ids"])[ww_mask]

    n_ww_sensors = int(ww_data["n_sites"])
    n_monitored = int(ww_monitored_subpops.shape[0])
    sensor_to_subpop = jnp.array(
        [int(ww_monitored_subpops[i % n_monitored]) for i in range(n_ww_sensors)]
    )
    ww_subpop_indices = sensor_to_subpop[ww_sensors]

    return {
        "hospital": {"obs": hosp_obs},
        "wastewater": {
            "obs": ww_conc,
            "times": ww_times,
            "subpop_indices": ww_subpop_indices,
            "sensor_indices": ww_sensors,
            "n_sensors": n_ww_sensors,
        },
    }


class TestDataAssembly:
    """Verify California datasets can be loaded and aligned to the model."""

    def test_hospital_data_shape(self, ca_hosp_data: DataDict) -> None:
        """
        Verify the CA hospital admissions dataset exposes expected keys.

        Parameters
        ----------
        ca_hosp_data
            California hospital admissions dictionary.
        """
        assert "daily_admits" in ca_hosp_data
        assert "population" in ca_hosp_data
        assert "n_days" in ca_hosp_data
        assert len(ca_hosp_data["daily_admits"]) >= N_DAYS_FIT
        assert float(ca_hosp_data["population"]) > 0
        assert ca_hosp_data["dates"][0] == START_DATE

    def test_wastewater_data_shape(self, ca_ww_data: DataDict) -> None:
        """
        Verify the CA wastewater dataset exposes expected keys and shapes.

        Parameters
        ----------
        ca_ww_data
            California wastewater dictionary.
        """
        for key in (
            "observed_conc",
            "site_ids",
            "time_indices",
            "n_sites",
            "n_obs",
        ):
            assert key in ca_ww_data
        n_obs = int(ca_ww_data["n_obs"])
        assert len(ca_ww_data["observed_conc"]) == n_obs
        assert len(ca_ww_data["site_ids"]) == n_obs
        assert len(ca_ww_data["time_indices"]) == n_obs

    def test_observation_padding(
        self,
        hw_model: MultiSignalModel,
        ca_hosp_data: DataDict,
    ) -> None:
        """
        Verify pad_observations produces the correct shape and NaN prefix.

        Parameters
        ----------
        hw_model
            Built hospital+wastewater model.
        ca_hosp_data
            California hospital admissions dictionary.
        """
        hosp_array = jnp.array(
            ca_hosp_data["daily_admits"][:N_DAYS_FIT], dtype=jnp.float32
        )
        padded = hw_model.pad_observations(hosp_array)
        n_init = hw_model.latent.n_initialization_points

        assert padded.shape[0] == n_init + N_DAYS_FIT
        assert jnp.all(jnp.isnan(padded[:n_init]))
        assert not jnp.any(jnp.isnan(padded[n_init:]))

    def test_shift_times_offsets_by_n_init(
        self,
        hw_model: MultiSignalModel,
        ca_ww_data: DataDict,
    ) -> None:
        """
        Verify shift_times offsets sparse time indices by n_init.

        Parameters
        ----------
        hw_model
            Built hospital+wastewater model.
        ca_ww_data
            California wastewater dictionary.
        """
        n_init = hw_model.latent.n_initialization_points
        raw_times = jnp.array(ca_ww_data["time_indices"][:10])
        shifted = hw_model.shift_times(raw_times)
        assert jnp.all(shifted == raw_times + n_init)

    def test_subpop_fractions_sum_to_one(self) -> None:
        """Verify the subpopulation fractions form a valid partition."""
        assert jnp.isclose(SUBPOP_FRACTIONS.sum(), 1.0)
        assert SUBPOP_FRACTIONS.shape[0] == 6

    def test_subpopulation_indices_reach_observation_process(
        self,
        hw_model: MultiSignalModel,
        ca_hosp_data: DataDict,
        ca_ww_data: DataDict,
    ) -> None:
        """Exercise subpopulation-index routing through MultiSignalModel.sample."""
        obs_data = _prepare_observation_data(
            hw_model,
            N_DAYS_FIT,
            jnp.array(ca_hosp_data["daily_admits"]),
            ca_ww_data,
            WW_MONITORED_SUBPOPS,
        )

        hw_model.validate_data(
            n_days_post_init=N_DAYS_FIT,
            subpop_fractions=SUBPOP_FRACTIONS,
            obs_start_date=START_DATE,
            **obs_data,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as model_trace:
                hw_model.sample(
                    n_days_post_init=N_DAYS_FIT,
                    population_size=float(ca_hosp_data["population"]),
                    subpop_fractions=SUBPOP_FRACTIONS,
                    obs_start_date=START_DATE,
                    **obs_data,
                )

        assert model_trace["wastewater_obs"]["is_observed"]
        assert (
            model_trace["wastewater_obs"]["value"].shape
            == obs_data["wastewater"]["subpop_indices"].shape
        )

    def test_subpopulation_counts_indices_reach_observation_process(
        self,
        hw_model: MultiSignalModel,
    ) -> None:
        """Regress subpopulation-index routing for issue 882's observation type."""
        ascertainment = IndependentAscertainment(
            name="regional_counts_ascertainment",
            rate_rvs={
                "regional_counts": DeterministicVariable(
                    "regional_ascertainment",
                    0.1,
                )
            },
        )
        observation = SubpopulationCounts(
            name="regional_counts",
            ascertainment_rate_rv=ascertainment.for_signal("regional_counts"),
            delay_distribution_rv=DeterministicPMF("regional_delay", jnp.array([1.0])),
            noise=PoissonNoise(),
        )
        model = MultiSignalModel(
            latent_process=hw_model.latent,
            observations={"regional_counts": observation},
            ascertainment_models={ascertainment.name: ascertainment},
        )
        subpop_indices = jnp.array([0, 2, 4])

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as model_trace:
                model.sample(
                    n_days_post_init=7,
                    population_size=1_000_000,
                    subpop_fractions=SUBPOP_FRACTIONS,
                    obs_start_date=START_DATE,
                    regional_counts={
                        "obs": None,
                        "subpop_indices": subpop_indices,
                    },
                )

        assert not model_trace["regional_counts_obs"]["is_observed"]
        assert model_trace["regional_counts_obs"]["value"].shape == (
            model.latent.n_initialization_points + 7,
            subpop_indices.shape[0],
        )


class TestModelFit:
    """Fit the H+W model and check posterior behavior."""

    @pytest.fixture(scope="class")
    def fitted_model(
        self,
        hw_model: MultiSignalModel,
        ca_hosp_data: DataDict,
        ca_ww_data: DataDict,
    ) -> MultiSignalModel:
        """
        Fit the H+W model to CA hospital and wastewater data via MCMC.

        Parameters
        ----------
        hw_model
            Built hospital+wastewater model.
        ca_hosp_data
            California hospital admissions dictionary.
        ca_ww_data
            California wastewater dictionary.

        Returns
        -------
        MultiSignalModel
            Model with MCMC results attached.
        """
        jax.clear_caches()

        obs_data = _prepare_observation_data(
            hw_model,
            N_DAYS_FIT,
            jnp.array(ca_hosp_data["daily_admits"]),
            ca_ww_data,
            WW_MONITORED_SUBPOPS,
        )
        population_size = float(ca_hosp_data["population"])

        hw_model.run(
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,
            rng_key=random.PRNGKey(42),
            mcmc_args={"num_chains": NUM_CHAINS, "progress_bar": False},
            n_days_post_init=N_DAYS_FIT,
            population_size=population_size,
            subpop_fractions=SUBPOP_FRACTIONS,
            obs_start_date=START_DATE,
            **obs_data,
        )

        samples = hw_model.mcmc.get_samples()
        jax.block_until_ready(samples)
        return hw_model

    @pytest.fixture(scope="class")
    def posterior_dt(self, fitted_model: MultiSignalModel) -> xr.DataTree:
        """
        Convert MCMC samples to an ArviZ DataTree, trimming the init period.

        Parameters
        ----------
        fitted_model
            Model with MCMC results.

        Returns
        -------
        xarray.DataTree
            ArviZ DataTree with the posterior group.
        """
        n_init = fitted_model.latent.n_initialization_points
        dt = az_base.from_numpyro(
            fitted_model.mcmc,
            dims={
                "latent_infections": ["time"],
                "latent_infections_by_subpop": ["time", "subpop"],
                "SubpopulationInfections::infections_aggregate": ["time"],
                "SubpopulationInfections::log_rt_baseline": ["rt_time", "dummy"],
                "SubpopulationInfections::rt_baseline": ["rt_time", "dummy"],
                "SubpopulationInfections::rt_subpop": ["rt_time", "subpop"],
                "hospital_predicted": ["time"],
            },
        )

        def trim_init(ds: xr.Dataset) -> xr.Dataset:
            """
            Trim the initialization period from the time dimension.

            Parameters
            ----------
            ds
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

    def test_mcmc_convergence(self, posterior_dt: xr.DataTree) -> None:
        """
        Check that core scalar parameters have acceptable Rhat and ESS.

        Parameters
        ----------
        posterior_dt
            ArviZ DataTree with the posterior group.
        """
        summary = az_stats.summary(
            posterior_dt,
            var_names=["I0", "log_rt_time_0"],
        )
        rhat = summary["r_hat"].astype(float)
        ess = summary["ess_bulk"].astype(float)
        assert (rhat < 1.1).all(), f"Rhat exceeded 1.1:\n{summary[rhat >= 1.1]}"
        assert (ess > 50).all(), f"ESS_bulk below 50:\n{summary[ess <= 50]}"

    def test_deviation_geometry(self, fitted_model: MultiSignalModel) -> None:
        """Five coordinate trajectories produce six correctly initialized regions."""
        samples = fitted_model.mcmc.get_samples()
        deviations = samples["SubpopulationInfections::subpop_deviations"]
        assert deviations.shape == (NUM_CHAINS * NUM_SAMPLES, N_DAYS_FIT, 6)
        np.testing.assert_allclose(deviations.sum(-1), 0, atol=1e-12)
        np.testing.assert_allclose(deviations[:, 0], 0, atol=1e-12)
        rt = samples["SubpopulationInfections::rt_subpop"]
        expected_initial = np.broadcast_to(
            samples["log_rt_time_0"][:, None], rt[:, 0].shape
        )
        np.testing.assert_allclose(np.log(rt[:, 0]), expected_initial, atol=1e-12)
        coordinates = {
            name: value
            for name, value in samples.items()
            if name.startswith("subpop_deviations_coordinates_")
        }
        assert coordinates
        assert all(value.shape[-1] == 5 for value in coordinates.values())
        if isinstance(
            fitted_model.latent.subpop_rt_deviation_process, WeeklyTemporalProcess
        ):
            weekly = samples["subpop_deviations_weekly"]
            assert weekly.shape[-1] == 6
            start_dow = fitted_model.latent.subpop_rt_deviation_process.start_dow
            offset = (START_DATE.weekday() - start_dow) % 7
            indices = (np.arange(N_DAYS_FIT) + offset) // 7
            np.testing.assert_allclose(deviations, weekly[:, indices])

    def test_deviation_diagnostics(
        self,
        fitted_model: MultiSignalModel,
        record_property: Callable[[str, object], None],
    ) -> None:
        """Record mixing diagnostics for deviation parameters and regional paths."""
        samples = fitted_model.mcmc.get_samples(group_by_chain=True)
        selected = {
            "deviations": samples["SubpopulationInfections::subpop_deviations"][
                :, :, [14, 42, 70]
            ],
            "rt": samples["SubpopulationInfections::rt_subpop"][:, :, [14, 42, 70]],
        }
        for name in ("deviation_autoreg", "deviation_innovation_sd"):
            if name in samples:
                selected[name] = samples[name]
        for name, diagnostics in summary(selected).items():
            for statistic in ("r_hat", "n_eff"):
                assert np.all(np.isfinite(diagnostics[statistic])), (name, diagnostics)
            record_property(f"{name}_max_rhat", float(np.max(diagnostics["r_hat"])))
            record_property(f"{name}_min_ess", float(np.min(diagnostics["n_eff"])))

    def test_wastewater_predictions(
        self,
        fitted_model: MultiSignalModel,
        ca_hosp_data: DataDict,
        ca_ww_data: DataDict,
    ) -> None:
        """Posterior predictions cover the tutorial wastewater measurements."""
        observations = _prepare_observation_data(
            fitted_model,
            N_DAYS_FIT,
            jnp.asarray(ca_hosp_data["daily_admits"]),
            ca_ww_data,
            WW_MONITORED_SUBPOPS,
        )
        wastewater = {
            key: value
            for key, value in observations["wastewater"].items()
            if key != "obs"
        }
        prediction = fitted_model.posterior_predictive(
            rng_key=jax.random.key(61),
            n_days_post_init=N_DAYS_FIT,
            population_size=float(ca_hosp_data["population"]),
            subpop_fractions=SUBPOP_FRACTIONS,
            obs_start_date=START_DATE,
            wastewater=wastewater,
        )
        values = np.asarray(prediction["wastewater_obs"])
        assert np.all(np.isfinite(values))
        low, high = np.quantile(values, [0.05, 0.95], axis=0)
        observed = np.asarray(observations["wastewater"]["obs"])
        assert np.mean((low <= observed) & (observed <= high)) >= 0.75

    def test_forecast_replay(
        self,
        fitted_model: MultiSignalModel,
        ca_hosp_data: DataDict,
        ca_ww_data: DataDict,
    ) -> None:
        """Forecasts retain fitted paths and extend both observation signals."""
        n_init = fitted_model.latent.n_initialization_points
        n_forecast = 14
        n_sensors = int(ca_ww_data["n_sites"])
        sensors = jnp.tile(jnp.arange(n_sensors), n_forecast)
        times = jnp.repeat(
            jnp.arange(N_DAYS_FIT, N_DAYS_FIT + n_forecast) + n_init, n_sensors
        )
        forecast = fitted_model.posterior_predictive(
            rng_key=jax.random.key(62),
            n_days_post_init=N_DAYS_FIT + n_forecast,
            population_size=float(ca_hosp_data["population"]),
            subpop_fractions=SUBPOP_FRACTIONS,
            obs_start_date=START_DATE,
            wastewater={
                "times": times,
                "subpop_indices": WW_MONITORED_SUBPOPS[
                    sensors % len(WW_MONITORED_SUBPOPS)
                ],
                "sensor_indices": sensors,
                "n_sensors": n_sensors,
            },
        )
        samples = fitted_model.mcmc.get_samples()
        for name, length in (
            ("SubpopulationInfections::rt_subpop", N_DAYS_FIT),
            ("SubpopulationInfections::subpop_deviations", N_DAYS_FIT),
            ("latent_infections_by_subpop", n_init + N_DAYS_FIT),
            ("hospital_predicted", n_init + N_DAYS_FIT),
            ("wastewater_predicted", n_init + N_DAYS_FIT),
        ):
            assert forecast[name].shape[1] == length + n_forecast
            np.testing.assert_allclose(
                forecast[name][:, :length], samples[name], rtol=1e-10, atol=1e-12
            )
            assert np.all(np.isfinite(forecast[name][:, -n_forecast:]))
        np.testing.assert_allclose(
            forecast["SubpopulationInfections::subpop_deviations"].sum(-1),
            0,
            atol=1e-12,
        )
        assert forecast["wastewater_obs"].shape == (
            NUM_CHAINS * NUM_SAMPLES,
            n_forecast * n_sensors,
        )
        assert np.all(np.isfinite(forecast["wastewater_obs"]))
        assert np.all(forecast["hospital_obs"][:, -n_forecast:] >= 0)

    def test_infection_trajectory_shape(self, posterior_dt: xr.DataTree) -> None:
        """
        Check the posterior infection trajectory has the expected shape
        and is strictly positive.

        Parameters
        ----------
        posterior_dt
            ArviZ DataTree with the posterior group.
        """
        aggregate = posterior_dt.posterior["latent_infections"]
        assert aggregate.sizes["time"] == N_DAYS_FIT
        assert (aggregate.values > 0).all()

        by_subpop = posterior_dt.posterior["latent_infections_by_subpop"]
        assert by_subpop.sizes["time"] == N_DAYS_FIT
        assert by_subpop.sizes["subpop"] == int(SUBPOP_FRACTIONS.shape[0])
        assert (by_subpop.values > 0).all()

    def test_rt_baseline_in_plausible_range(self, posterior_dt: xr.DataTree) -> None:
        """
        Check the posterior median baseline R(t) falls in a plausible
        epidemiological range.

        Parameters
        ----------
        posterior_dt
            ArviZ DataTree with the posterior group.
        """
        rt_baseline = posterior_dt.posterior["SubpopulationInfections::rt_baseline"]
        rt_median = rt_baseline.median(dim=["chain", "draw"]).values
        rt_median = np.squeeze(rt_median)
        assert rt_median.shape[0] == N_DAYS_FIT
        n_stable = N_DAYS_FIT - N_DAYS_TAIL_TRIM
        stable = rt_median[:n_stable]
        assert (stable > 0.3).all()
        assert (stable < 4.0).all()

    def test_hospital_predictions_track_observations(
        self,
        posterior_dt: xr.DataTree,
        ca_hosp_data: DataDict,
    ) -> None:
        """
        Check posterior predicted hospital admissions are positive and
        on the same order of magnitude as observed admissions.

        Parameters
        ----------
        posterior_dt
            ArviZ DataTree with the posterior group.
        ca_hosp_data
            California hospital admissions dictionary.
        """
        hosp_pred = posterior_dt.posterior["hospital_predicted"]
        assert hosp_pred.sizes["time"] == N_DAYS_FIT

        n_stable = N_DAYS_FIT - N_DAYS_TAIL_TRIM
        hosp_pred_stable = hosp_pred.isel(time=slice(0, n_stable))
        hosp_pred_median = float(
            hosp_pred_stable.median(dim=["chain", "draw", "time"]).values
        )
        hosp_obs_mean = float(
            np.mean(np.asarray(ca_hosp_data["daily_admits"])[:n_stable])
        )

        assert hosp_pred_median > 0
        assert hosp_obs_mean / 10 <= hosp_pred_median <= hosp_obs_mean * 10, (
            f"pred median (first {n_stable} days)={hosp_pred_median}, "
            f"obs mean={hosp_obs_mean}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
