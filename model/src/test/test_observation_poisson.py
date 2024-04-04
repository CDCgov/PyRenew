# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import numpyro as npro
from pyrenew.observation import PoissonObservation


def test_poisson_obs():
    """
    Check that an PoissonObservation can sample
    """

    pois = PoissonObservation("rv")

    np.random.seed(223)
    rates = np.random.randint(1, 5, size=10)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_pois, *_ = pois.sample(predicted=rates)

    testing.assert_array_equal(sim_pois, jnp.ceil(sim_pois))
