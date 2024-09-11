# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
import pytest
from numpy.testing import assert_almost_equal

from pyrenew.process import ARProcess


def test_ar_can_be_sampled():
    """
    Check that an AR process
    can be initialized and sampled from,
    and that output shapes are as expected.
    """
    ar = ARProcess()
    with numpyro.handlers.seed(rng_seed=62):
        # can sample
        ar(
            noise_name="ar1process_noise",
            n=3532,
            init_vals=jnp.array([50.0]),
            autoreg=jnp.array([0.95]),
            noise_sd=0.5,
        )

    with numpyro.handlers.seed(rng_seed=62):
        res1, *_ = ar(
            noise_name="ar3process_noise",
            n=1230,
            init_vals=jnp.array([50.0, 49.9, 48.2]),
            autoreg=jnp.array([0.05, 0.025, 0.025]),
            noise_sd=0.5,
        )
        res2, *_ = ar(
            noise_name="ar3process_noise",
            n=1230,
            init_vals=jnp.array([50.0, 49.9, 48.2]),
            autoreg=jnp.array([0.05, 0.025, 0.025]),
            noise_sd=[0.25],
        )
        res3, *_ = ar(
            noise_name="ar3process_noise",
            n=1230,
            init_vals=jnp.array([50.0, 49.9, 48.2]),
            autoreg=jnp.array([0.05, 0.025, 0.025]),
            noise_sd=jnp.array([0.25]),
        )

        assert jnp.shape(res1.value) == jnp.shape(res2.value)
        assert jnp.shape(res2.value) == jnp.shape(res3.value)
        assert jnp.shape(res3.value) == (1230,)


def test_ar_shape_validation():
    """
    Test that AR process sample() method validates
    the shapes of its inputs as expected.
    """
    # vector valued noise raises
    # error
    ar = ARProcess()

    with pytest.raises(ValueError, match="must be a scalar"):
        ar(
            noise_name="ar3process_noise",
            n=1230,
            init_vals=jnp.array([50.0, 49.9, 48.2]),
            autoreg=jnp.array([0.05, 0.025, 0.025]),
            noise_sd=jnp.array([1.0, 2.0]),
        )
    with pytest.raises(ValueError, match="must be a scalar"):
        ar(
            noise_name="ar3process_noise",
            n=1230,
            init_vals=jnp.array([50.0, 49.9, 48.2]),
            autoreg=jnp.array([0.05, 0.025, 0.025]),
            noise_sd=[1.0, 2.0],
        )
    # bad dimensionality raises error
    with pytest.raises(ValueError, match="Initial values array"):
        ar(
            noise_name="ar3process_noise",
            n=1230,
            init_vals=jnp.array([50.0, 49.9, 48.2]),
            autoreg=jnp.array([[0.05, 0.025, 0.025]]),
            noise_sd=0.5,
        )
    with pytest.raises(ValueError, match="Initial values array"):
        ar(
            noise_name="ar3process_noise",
            n=1230,
            init_vals=jnp.array([[50.0, 49.9, 48.2]]),
            autoreg=jnp.array([0.05, 0.025, 0.025]),
            noise_sd=0.5,
        )
    with pytest.raises(ValueError, match="Initial values array"):
        ar(
            noise_name="ar3process_noise",
            n=1230,
            init_vals=jnp.array([50.0, 49.9, 1, 1, 1]),
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
    ar = ARProcess()
    with numpyro.handlers.seed(rng_seed=62):
        # check it regresses to mean
        # when started away from it
        long_ts, *_ = ar(
            noise_name="arprocess_noise",
            n=10000,
            init_vals=ar_inits,
            autoreg=jnp.array([0.75]),
            noise_sd=noise_sd,
        )
        assert_almost_equal(long_ts.value[0], ar_inits)
        assert jnp.abs(long_ts.value[-1]) < 4 * noise_sd
