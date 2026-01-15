# numpydoc ignore=GL08

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import numpyro

from pyrenew.observation import PoissonObservation


def test_poisson_obs():
    """
    Check that an PoissonObservation can sample
    """

    pois = PoissonObservation("rv")

    rates = np.random.randint(1, 5, size=10)
    with numpyro.handlers.seed(rng_seed=223):
        sim_pois = pois(mu=rates)

    testing.assert_array_equal(sim_pois, jnp.ceil(sim_pois))


def test_poisson_validate():
    """
    Check that PoissonObservation.validate() runs without error.
    """
    PoissonObservation.validate()
