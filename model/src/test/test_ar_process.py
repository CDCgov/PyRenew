# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
from numpy.testing import assert_almost_equal
from pyrenew.process import ARProcess


def test_ar_can_be_sampled():
    """
    Check that an AR process
    can be initialized and sampled from
    """
    ar1 = ARProcess(5, jnp.array([0.95]), jnp.array([0.5]))
    with numpyro.handlers.seed(rng_seed=62):
        # can sample with and without inits
        ar1.sample(3532, inits=jnp.array([50.0]))
        ar1.sample(5023)

    ar3 = ARProcess(5, jnp.array([0.05, 0.025, 0.025]), jnp.array([0.5]))
    with numpyro.handlers.seed(rng_seed=62):
        # can sample with and without inits
        ar3.sample(1230)
        ar3.sample(52, inits=jnp.array([50.0, 49.9, 48.2]))


def test_ar_samples_correctly_distributed():
    """
    Check that AR processes have correctly-
    distributed steps.
    """
    ar_mean = 5
    noise_sd = jnp.array([0.5])
    ar_inits = jnp.array([25.0])
    ar1 = ARProcess(ar_mean, jnp.array([0.75]), noise_sd)
    with numpyro.handlers.seed(rng_seed=62):
        # check it regresses to mean
        # when started away from it
        long_ts, *_ = ar1.sample(10000, inits=ar_inits)
        assert_almost_equal(long_ts.array[0], ar_inits)
        assert jnp.abs(long_ts.array[-1] - ar_mean) < 4 * noise_sd

