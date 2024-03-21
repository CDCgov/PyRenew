# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np
import numpyro as npro
import polars as pl
from pyrenew.latent import HospitalAdmissions, Infections
from pyrenew.model import HospitalizationsModel
from pyrenew.observation import PoissonObservation
from pyrenew.process import RtRandomWalkProcess


def test_model_hosp_no_obs_model():
    """
    Checks that the partially deterministic
    Hospitalization model runs
    """

    latent_infections = Infections(jnp.array([0.25, 0.25, 0.25, 0.25]))
    Rt_process = RtRandomWalkProcess()
    latent_hospitalizations = HospitalAdmissions(
        inf_hosp_int=jnp.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.1, 0.1, 0.05],
        ),
        infections_varname="infections",
        hospitalizations_predicted_varname="observed_hospitalizations",
    )

    model0 = HospitalizationsModel(
        Rt_process=Rt_process,
        latent_infections=latent_infections,
        latent_hospitalizations=latent_hospitalizations,
        observed_hospitalizations=None,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model0_samp = model0.sample(constants={"n_timepoints": 30})

    model0.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        random_variables=dict(observed_hospitalizations=model0_samp.sampled),
        constants=dict(n_timepoints=30),
    )

    inf = model0.spread_draws(["observed_hospitalizations"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("observed_hospitalizations").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500


def test_model_hosp_with_obs_model():
    """
    Checks that the random Hospitalization model runs
    """

    latent_infections = Infections(jnp.array([0.25, 0.25, 0.25, 0.25]))
    Rt_process = RtRandomWalkProcess()
    observed_hospitalizations = PoissonObservation(
        rate_varname="latent",
        counts_varname="observed_hospitalizations",
    )

    latent_hospitalizations = HospitalAdmissions(
        inf_hosp_int=jnp.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.1, 0.1, 0.05],
        ),
        infections_varname="infections",
    )

    model1 = HospitalizationsModel(
        Rt_process=Rt_process,
        latent_infections=latent_infections,
        latent_hospitalizations=latent_hospitalizations,
        observed_hospitalizations=observed_hospitalizations,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.sample(constants={"n_timepoints": 30})

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        random_variables=dict(
            observed_hospitalizations=model1_samp.sampled,
            infections=model1_samp.infections,
        ),
        constants=dict(n_timepoints=30),
    )

    inf = model1.spread_draws(["predicted_hospitalizations"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("predicted_hospitalizations").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500
