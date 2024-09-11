# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from pyrenew.process import ARProcess


@pytest.mark.parametrize(
    ["init_vals", "autoreg", "noise_sd", "n"],
    [
        # AR1, 1D
        [jnp.array([50.0]), jnp.array([0.95]), 0.5, 1353],
        # AR1, 1D, length 1
        [jnp.array([50.0]), jnp.array([0.95]), 0.5, 1],
        # AR1, multi-dim
        [
            jnp.array([[43.1, -32.5, 3.2, -0.5]]).reshape((1, 4)),
            jnp.array([[0.50, 0.205, 0.232, 0.25]]).reshape((1, 4)),
            0.73,
            5322,
        ],
        # AR3, one dim
        [
            jnp.array([[43.1, -32.5, 0.52]]).reshape((3, -1)),
            jnp.array([[0.50, 0.205, 0.25]]).reshape((3, -1)),
            0.802,
            532,
        ],
    ],
)
def test_ar_can_be_sampled(init_vals, autoreg, noise_sd, n):
    """
    Check that an AR process
    can be initialized and sampled from,
    and that output shapes are as expected.
    """
    ar = ARProcess()
    with numpyro.handlers.seed(rng_seed=62):
        # can sample

        res = ar(
            noise_name="ar3process_noise",
            n=n,
            init_vals=init_vals,
            autoreg=autoreg,
            noise_sd=noise_sd,
        )
        order = jnp.shape(autoreg)[0]
        non_time_dims = jnp.shape(jnp.atleast_1d(autoreg))[1:]
        non_time_dims = tuple(x for x in non_time_dims if x != 1)
        expected_shape = (n,) + non_time_dims
        assert jnp.shape(res) == expected_shape
        assert_array_almost_equal(
            jnp.squeeze(res[0:order, ...]), jnp.squeeze(init_vals)
        )


@pytest.mark.parametrize(
    ["init_vals", "autoreg", "noise_sd", "n"],
    [
        # autoreg higher dim than init vals
        [
            jnp.array([50.0, 49.9, 48.2]),
            jnp.array([[0.05, 0.025, 0.025]]),
            0.5,
            1230,
        ],
        # init vals higher dim than autoreg
        [
            jnp.array([[50.0, 49.9, 48.2]]),
            jnp.array([0.05, 0.025, 0.025]),
            0.5,
            1230,
        ],
    ],
)
def test_ar_shape_validation(init_vals, autoreg, noise_sd, n):
    """
    Test that AR process sample() method validates
    the shapes of its inputs as expected.
    """
    # vector valued noise raises
    # error
    ar = ARProcess()

    # bad dimensionality raises error
    with pytest.raises(ValueError, match="Initial values array"):
        ar(
            noise_name="test_ar_noise",
            n=n,
            init_vals=init_vals,
            autoreg=autoreg,
            noise_sd=noise_sd,
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
        long_ts = ar(
            noise_name="arprocess_noise",
            n=10000,
            init_vals=ar_inits,
            autoreg=jnp.array([0.75]),
            noise_sd=noise_sd,
        )
        assert_almost_equal(long_ts[0], ar_inits)
        assert jnp.abs(long_ts[-1]) < 4 * noise_sd
