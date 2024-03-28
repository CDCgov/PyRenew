# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np
import numpyro as npro
import polars as pl
from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import Infections
from pyrenew.model import RtInfectionsRenewalModel
from pyrenew.observation import PoissonObservation


def test_model_basicrenewal_no_obs_model():
    """
    Test the basic semi-deterministic renewal model runs. Semi-deterministic
    from the perspective of the infections. It returns expected, not sampled.
    """

    gen_int = DeterministicPMF(
        (jnp.array([0.25, 0.25, 0.25, 0.25]),),
    )

    latent_infections = Infections(gen_int=gen_int)

    model0 = RtInfectionsRenewalModel(latent_infections=latent_infections)

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model0_samp = model0.sample(constants={"n_timepoints": 30})

    model0.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        random_variables=dict(observed_infections=model0_samp.observed),
        constants=dict(n_timepoints=30),
    )

    inf = model0.spread_draws(["latent_infections"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("latent_infections").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500


def test_model_basicrenewal_with_obs_model():
    """
    Test the basic random renewal model runs. Random
    from the perspective of the infections. It returns sampled, not expected.
    """

    gen_int = DeterministicPMF(
        (jnp.array([0.25, 0.25, 0.25, 0.25]),),
    )

    latent_infections = Infections(gen_int=gen_int)

    observed_infections = PoissonObservation(
        rate_varname="latent",
        counts_varname="observed_infections",
    )

    model1 = RtInfectionsRenewalModel(
        latent_infections=latent_infections,
        observed_infections=observed_infections,
    )

    # Sampling and fitting model 1 (with obs infections)
    np.random.seed(2203)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.sample(constants={"n_timepoints": 30})

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(22),
        random_variables=dict(observed_infections=model1_samp.observed),
        constants=dict(n_timepoints=30),
    )

    inf = model1.spread_draws(["latent_infections"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("latent_infections").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500
