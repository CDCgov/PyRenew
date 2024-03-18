import jax
import jax.numpy as jnp
import numpy as np
import numpyro as npro
import polars as pl
from pyrenew.models import HospitalizationsModel
from pyrenew.observations import (
    HospitalizationsObservation,
    InfectionsObservation,
    PoissonObservation,
)
from pyrenew.processes import RtRandomWalkProcess


def test_model_hosp_no_obs_model():
    """
    Checks that the partially deterministic
    Hospitalization model runs
    """

    infections_obs = InfectionsObservation(jnp.array([0.25, 0.25, 0.25, 0.25]))
    Rt_process = RtRandomWalkProcess()
    hosp_obs = HospitalizationsObservation(
        inf_hosp_int=jnp.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.1, 0.1, 0.05],
        ),
        infections_obs_varname="infections",
    )

    model0 = HospitalizationsModel(
        Rt_process=Rt_process, infections_obs=infections_obs, hosp_obs=hosp_obs
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model0_samp = model0.model(constants={"n_timepoints": 30})

    model0.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        random_variables=dict(hospitalizations_obs=model0_samp.samp_hosp),
        constants=dict(n_timepoints=30),
    )

    inf = model0.spread_draws(["hospitalizations_predicted"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("hospitalizations_predicted").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500


def test_model_hosp_with_obs_model():
    """
    Checks that the random Hospitalization model runs
    """

    infections_obs = InfectionsObservation(jnp.array([0.25, 0.25, 0.25, 0.25]))
    Rt_process = RtRandomWalkProcess()
    hosp_obs = HospitalizationsObservation(
        inf_hosp_int=jnp.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.1, 0.1, 0.05],
        ),
        infections_obs_varname="infections",
        hosp_dist=PoissonObservation(
            rate_varname="hospitalizations_predicted",
            counts_varname="hospitalizations_obs",
        ),
    )

    model1 = HospitalizationsModel(
        Rt_process=Rt_process, infections_obs=infections_obs, hosp_obs=hosp_obs
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.model(constants={"n_timepoints": 30})

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        random_variables=dict(hospitalizations_obs=model1_samp.samp_hosp),
        constants=dict(n_timepoints=30),
    )

    inf = model1.spread_draws(["hospitalizations_predicted"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("hospitalizations_predicted").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500
