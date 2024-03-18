import jax
import jax.numpy as jnp
import numpy as np
import numpyro as npro
import polars as pl
from pyrenew.models import BasicRenewalModel
from pyrenew.observations import InfectionsObservation, PoissonObservation


def test_model_basicrenewal_no_obs_model():
    """
    Test the basic semi-deterministic renewal model runs. Semi-deterministic
    from the perspective of the infections. It returns expected, not sampled.
    """

    infections_obs0 = InfectionsObservation(
        gen_int=jnp.array([0.25, 0.25, 0.25, 0.25]),
    )

    model0 = BasicRenewalModel(infections_obs=infections_obs0)

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model0_samp = model0.model(constants={"n_timepoints": 30})

    model0.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        random_variables=dict(infections_obs=model0_samp.infect_observed),
        constants=dict(n_timepoints=30),
    )

    inf = model0.spread_draws(["infections_mean"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("infections_mean").mean())
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

    infections_obs1 = InfectionsObservation(
        gen_int=jnp.array([0.25, 0.25, 0.25, 0.25]),
        inf_observation_model=PoissonObservation(
            rate_varname="infections_mean",
            counts_varname="infections_obs",
        ),
    )

    model1 = BasicRenewalModel(infections_obs=infections_obs1)

    # Sampling and fitting model 1 (with obs infections)
    np.random.seed(2203)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.model(constants={"n_timepoints": 30})

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(22),
        random_variables=dict(infections_obs=model1_samp.infect_observed),
        constants=dict(n_timepoints=30),
    )

    inf = model1.spread_draws(["infections_mean"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("infections_mean").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500
