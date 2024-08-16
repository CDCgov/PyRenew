# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import numpyro
from pyrenew.observation import PoissonObservation


def test_poisson_obs():
    """
    Check that an PoissonObservation can sample.
    """

    pois = PoissonObservation("rv")

    rates = np.random.randint(1, 5, size=10)
    with numpyro.handlers.seed(rng_seed=223):
        sim_pois, *_ = pois(mu=rates)

    testing.assert_array_equal(sim_pois.value, jnp.ceil(sim_pois.value))


def test_poisson_clipping():
    """
    Check that the clipping of the mean parameter works correctly.
    """

    pois = PoissonObservation(name="pois_rv")

    small_mu = 1e-10
    expected_clipped_mu = jnp.clip(
        small_mu + jnp.finfo(float).eps, min=jnp.finfo(float).eps, max=jnp.inf
    )

    with numpyro.handlers.seed(rng_seed=223):
        sim_pois, *_ = pois(mu=small_mu)

    mean_sample_value = jnp.mean(sim_pois.value)
    testing.assert_array_almost_equal(
        mean_sample_value,
        expected_clipped_mu,
        decimal=5,
    )
