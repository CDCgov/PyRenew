"""
Integration test: weekly hospital + daily ED model with joint ascertainment.

This exercises the mixed-cadence H+E structure with a ``JointAscertainment``
model shared by the hospital and ED visit observation processes.
"""

from __future__ import annotations

from datetime import date

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import polars as pl
import pytest

from pyrenew.ascertainment import AscertainmentSignal
from pyrenew.model import MultiSignalModel
from pyrenew.time import MMWR_WEEK

pytestmark = pytest.mark.integration


N_DAYS_FIT = 126
NUM_WARMUP = 500
NUM_SAMPLES = 500
NUM_CHAINS = 4
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


class TestJointStructure:
    """Check that the fixture has the intended H+E joint structure."""

    def test_observation_cadences(
        self,
        he_weekly_joint_ascertainment_model: MultiSignalModel,
    ) -> None:
        """
        Verify hospital is weekly MMWR and ED visits remain daily.

        Parameters
        ----------
        he_weekly_joint_ascertainment_model : MultiSignalModel
            Built model with joint ascertainment.
        """
        model = he_weekly_joint_ascertainment_model

        hospital = model.observations["hospital"]
        assert hospital.aggregation == "weekly"
        assert hospital.reporting_schedule == "regular"
        assert hospital.start_dow == MMWR_WEEK

        ed_visits = model.observations["ed_visits"]
        assert ed_visits.aggregation == "daily"
        assert ed_visits.day_of_week_rv is not None

    def test_joint_ascertainment_is_registered(
        self,
        he_weekly_joint_ascertainment_model: MultiSignalModel,
    ) -> None:
        """
        Verify both count observations use accessors from the same model.

        Parameters
        ----------
        he_weekly_joint_ascertainment_model : MultiSignalModel
            Built model with joint ascertainment.
        """
        model = he_weekly_joint_ascertainment_model
        assert set(model.ascertainment_models) == {"he_ascertainment"}

        ascertainment = model.ascertainment_models["he_ascertainment"]
        assert ascertainment.signals == ("hospital", "ed_visits")

        hospital_rate = model.observations["hospital"].ascertainment_rate_rv
        ed_rate = model.observations["ed_visits"].ascertainment_rate_rv
        assert isinstance(hospital_rate, AscertainmentSignal)
        assert isinstance(ed_rate, AscertainmentSignal)
        assert hospital_rate.ascertainment_name == "he_ascertainment"
        assert hospital_rate.signal_name == "hospital"
        assert ed_rate.ascertainment_name == "he_ascertainment"
        assert ed_rate.signal_name == "ed_visits"

    def test_weekly_obs_alignment(
        self,
        he_weekly_joint_ascertainment_model: MultiSignalModel,
        weekly_hosp: pl.DataFrame,
    ) -> None:
        """
        Verify weekly hospital observations align to the dense period grid.

        Parameters
        ----------
        he_weekly_joint_ascertainment_model : MultiSignalModel
            Built model with joint ascertainment.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        """
        model = he_weekly_joint_ascertainment_model
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
    """Check the NumPyro graph for joint ascertainment and mixed cadence."""

    def test_joint_ascertainment_sites_are_sampled_once(
        self,
        he_weekly_joint_ascertainment_model: MultiSignalModel,
        weekly_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> None:
        """
        Single model execution exposes one joint latent vector and two rates.

        Parameters
        ----------
        he_weekly_joint_ascertainment_model : MultiSignalModel
            Built model with joint ascertainment.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.
        """
        model = he_weekly_joint_ascertainment_model
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
        assert trace["he_ascertainment_eta"]["type"] == "sample"
        assert trace["he_ascertainment_eta"]["value"].shape == (2,)
        assert trace["he_ascertainment_hospital"]["type"] == "deterministic"
        assert trace["he_ascertainment_ed_visits"]["type"] == "deterministic"
        assert trace["hospital_predicted_daily"]["value"].shape == (n_total,)
        assert trace["ed_visits_predicted"]["value"].shape == (n_total,)


class TestModelFit:
    """Fit the joint-ascertainment H+E model and check core outputs."""

    @pytest.fixture(scope="class")
    def fitted_model(
        self,
        he_weekly_joint_ascertainment_model: MultiSignalModel,
        weekly_hosp: pl.DataFrame,
        daily_ed: pl.DataFrame,
    ) -> MultiSignalModel:
        """
        Fit the mixed-cadence joint-ascertainment H+E model.

        Parameters
        ----------
        he_weekly_joint_ascertainment_model : MultiSignalModel
            Built model with joint ascertainment.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        daily_ed : pl.DataFrame
            Daily ED visits.

        Returns
        -------
        MultiSignalModel
            Model with MCMC results attached.
        """
        model = he_weekly_joint_ascertainment_model
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

        model.run(
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,
            rng_key=random.PRNGKey(42),
            mcmc_args={"num_chains": NUM_CHAINS, "progress_bar": False},
            n_days_post_init=N_DAYS_FIT,
            population_size=float(weekly_hosp["pop"][0]),
            obs_start_date=OBS_START_DATE,
            hospital={"obs": hosp_obs},
            ed_visits={"obs": ed_obs},
        )

        samples = model.mcmc.get_samples()
        jax.block_until_ready(samples)
        return model

    @pytest.fixture(scope="class")
    def posterior_dt(
        self,
        fitted_model: MultiSignalModel,
    ):
        """
        Convert MCMC samples to an ArviZ DataTree, trimming init days.

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
                "he_ascertainment_eta": ["signal"],
                "latent_infections": ["time"],
                "PopulationInfections::infections_aggregate": ["time"],
                "PopulationInfections::log_rt_single": ["time", "dummy"],
                "PopulationInfections::rt_single": ["time", "dummy"],
                "hospital_predicted_daily": ["time"],
                "hospital_predicted": ["week"],
                "ed_visits_predicted": ["time"],
            },
        )

        def trim_init(ds):
            """
            Trim the initialization period from daily-time variables.

            Parameters
            ----------
            ds
                Dataset to trim.

            Returns
            -------
            xarray.Dataset
                Dataset with ``time`` sliced to ``[n_init:]``.
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
        Check that core scalar parameters have acceptable Rhat and ESS.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        """
        summary = az.summary(
            posterior_dt,
            var_names=["I0", "log_rt_time_0", "he_ascertainment_eta"],
        )
        rhat = summary["r_hat"].astype(float)
        ess = summary["ess_bulk"].astype(float)
        assert (rhat < 1.05).all(), f"Rhat exceeded 1.05:\n{summary[rhat >= 1.05]}"
        assert (ess > 100).all(), f"ESS_bulk below 100:\n{summary[ess <= 100]}"

    def test_joint_rates_are_in_posterior(
        self,
        posterior_dt,
        true_params: dict,
    ) -> None:
        """
        Check signal-specific rates are recorded and have plausible scale.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        true_params : dict
            Ground-truth parameter dictionary.
        """
        posterior = posterior_dt.posterior
        assert "he_ascertainment_hospital" in posterior
        assert "he_ascertainment_ed_visits" in posterior

        true_ihr = true_params["hospitalizations"]["ihr"]
        true_iedr = true_params["ed_visits"]["iedr"]
        ihr_median = float(
            posterior["he_ascertainment_hospital"].median(dim=["chain", "draw"]).values
        )
        iedr_median = float(
            posterior["he_ascertainment_ed_visits"].median(dim=["chain", "draw"]).values
        )

        assert true_ihr / 5 <= ihr_median <= true_ihr * 5
        assert true_iedr / 5 <= iedr_median <= true_iedr * 5

    def test_prediction_shapes(
        self,
        posterior_dt,
        weekly_hosp: pl.DataFrame,
    ) -> None:
        """
        Check predictions live on weekly hospital and daily ED grids.

        Parameters
        ----------
        posterior_dt : xarray.DataTree
            ArviZ DataTree with posterior group.
        weekly_hosp : pl.DataFrame
            Weekly hospital admissions.
        """
        posterior = posterior_dt.posterior
        assert posterior["latent_infections"].sizes["time"] == N_DAYS_FIT
        assert posterior["hospital_predicted_daily"].sizes["time"] == N_DAYS_FIT
        assert posterior["hospital_predicted"].sizes["week"] >= len(weekly_hosp)
        assert posterior["ed_visits_predicted"].sizes["time"] == N_DAYS_FIT


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
