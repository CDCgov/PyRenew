# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro as npro
import numpyro.distributions as dist
import polars as pl
import pytest
from pyrenew.deterministic import DeterministicPMF, NullObservation
from pyrenew.latent import Infections, Infections0
from pyrenew.model import RtInfectionsRenewalModel
from pyrenew.observation import PoissonObservation
from pyrenew.process import RtRandomWalkProcess


def test_model_basicrenewal_no_obs_model():
    """
    Test the basic semi-deterministic renewal model runs. Semi-deterministic
    from the perspective of the infections. It returns expected, not sampled.
    """

    gen_int = DeterministicPMF(jnp.array([0.25, 0.25, 0.25, 0.25]))

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()

    rt = RtRandomWalkProcess()

    model0 = RtInfectionsRenewalModel(
        gen_int=gen_int,
        I0=I0,
        latent_infections=latent_infections,
        Rt_process=rt,
        # Explicitly use None, this should call the NullObservation
        observation_process=None,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model0_samp = model0.sample(n_timepoints=30)

    # Generating
    model0.observation_process = NullObservation()
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model0.sample(n_timepoints=30)

    np.testing.assert_array_equal(model0_samp.Rt, model1_samp.Rt)
    np.testing.assert_array_equal(
        model0_samp.latent_infections, model1_samp.latent_infections
    )
    np.testing.assert_array_equal(
        model0_samp.sampled_infections, model1_samp.sampled_infections
    )

    model0.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        observed_infections=model0_samp.sampled_infections,
        n_timepoints=30,
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

    gen_int = DeterministicPMF(jnp.array([0.25, 0.25, 0.25, 0.25]))

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()

    observed_infections = PoissonObservation()

    rt = RtRandomWalkProcess()

    model1 = RtInfectionsRenewalModel(
        I0=I0,
        gen_int=gen_int,
        latent_infections=latent_infections,
        observation_process=observed_infections,
        Rt_process=rt,
    )

    # Sampling and fitting model 1 (with obs infections)
    np.random.seed(2203)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.sample(n_timepoints=30)

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(22),
        observed_infections=model1_samp.sampled_infections,
        n_timepoints=30,
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


@pytest.mark.mpl_image_compare
def test_model_basicrenewal_plot() -> plt.Figure:
    """
    Check that the posterior sample looks the same (reproducibility)

    Returns
    -------
    plt.Figure
        The figure object

    Notes
    -----
    IMPORTANT: If this test fails, it may be that you need
    to regenerate the figures. To do so, you can the test using the following
    command:

      poetry run pytest --mpl-generate-path=src/test/baseline

    This will skip validating the figure and save the new figure in the
    `src/test/baseline` folder.
    """
    gen_int = DeterministicPMF(jnp.array([0.25, 0.25, 0.25, 0.25]))

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()

    observed_infections = PoissonObservation()

    rt = RtRandomWalkProcess()

    model1 = RtInfectionsRenewalModel(
        I0=I0,
        gen_int=gen_int,
        latent_infections=latent_infections,
        observation_process=observed_infections,
        Rt_process=rt,
    )

    # Sampling and fitting model 1 (with obs infections)
    np.random.seed(2203)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.sample(n_timepoints=30)

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(22),
        observed_infections=model1_samp.sampled_infections,
        n_timepoints=30,
    )

    return model1.plot_posterior(
        var="latent_infections",
        obs_signal=model1_samp.sampled_infections,
    )


def test_model_basicrenewal_padding() -> None:  # numpydoc ignore=GL08
    gen_int = DeterministicPMF(jnp.array([0.25, 0.25, 0.25, 0.25]))

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()

    observed_infections = PoissonObservation()

    rt = RtRandomWalkProcess()

    model1 = RtInfectionsRenewalModel(
        I0=I0,
        gen_int=gen_int,
        latent_infections=latent_infections,
        observation_process=observed_infections,
        Rt_process=rt,
    )

    # Sampling and fitting model 1 (with obs infections)
    np.random.seed(2203)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.sample(n_timepoints=30)

    new_obs = jnp.hstack(
        [jnp.repeat(jnp.nan, 5), model1_samp.sampled_infections[5:]],
    )

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(22),
        observed_infections=new_obs,
        n_timepoints=30,
        padding=5,
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
