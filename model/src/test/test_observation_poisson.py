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

    testing.assert_array_equal(sim_pois, jnp.ceil(sim_pois))


def test_poisson_obs_eps():
    """
    Check that `eps` default for PoissonObservation does not
    fail when `eps = 0` does.
    """

    pois_eps_default = PoissonObservation(name="rv", eps=jnp.finfo(float).eps)
    pois_eps_zero = PoissonObservation(name="rv", eps=0.0)

    rates = np.random.randint(1, 5, size=10)

    # eps = jnp.finfo(float).eps does not fail
    with numpyro.handlers.seed(rng_seed=223):
        sim_pois_eps_default, *_ = pois_eps_default(mu=rates)
    testing.assert_array_equal(
        sim_pois_eps_default, jnp.ceil(sim_pois_eps_default)
    )

    # esp = 0.0 fails?
    # with pytest.raises(AssertionError):
    with numpyro.handlers.seed(rng_seed=223):
        sim_pois_eps_zero, *_ = pois_eps_zero(mu=rates)
    testing.assert_array_equal(sim_pois_eps_zero, jnp.ceil(sim_pois_eps_zero))
