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
    ar1 = ARProcess("ar1process")
    with numpyro.handlers.seed(rng_seed=62):
        # can sample
        ar1(
            n=3532,
            init_vals=jnp.array([50.0]),
            autoreg=jnp.array([0.95]),
            noise_sd=0.5,
        )

    ar3 = ARProcess("ar3process")

    with numpyro.handlers.seed(rng_seed=62):
        # can sample
        ar3(
            n=1230,
            init_vals=jnp.array([50.0, 49.9, 48.2]),
            autoreg=jnp.array([0.05, 0.025, 0.025]),
            noise_sd=0.5,
        )


def test_ar_samples_correctly_distributed():
    """
    Check that AR processes have correctly-
    distributed steps.
    """
    noise_sd = jnp.array([0.5])
    ar_inits = jnp.array([25.0])
    ar = ARProcess("arprocess")
    with numpyro.handlers.seed(rng_seed=62):
        # check it regresses to mean
        # when started away from it
        long_ts, *_ = ar(
            n=10000,
            init_vals=ar_inits,
            autoreg=jnp.array([0.75]),
            noise_sd=noise_sd,
        )
        assert_almost_equal(long_ts.value[0], ar_inits)
        assert jnp.abs(long_ts.value[-1]) < 4 * noise_sd
