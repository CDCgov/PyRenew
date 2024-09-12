# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
import pytest
from numpy.testing import assert_array_almost_equal

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
            jnp.array([0.73]),
            5322,
        ],
        # AR3, one dim
        [
            jnp.array([43.1, -32.5, 0.52]),
            jnp.array([0.50, 0.205, 0.25]),
            jnp.array(0.802),
            6432,
        ],
        # AR3, one dim but unsqueezed
        [
            jnp.array([[43.1, -32.5, 0.52]]).reshape((3, -1)),
            jnp.array([[0.50, 0.205, 0.25]]).reshape((3, -1)),
            0.802,
            6432,
        ],
        # AR3, two sets of inits
        # one set of AR coefficients
        [
            jnp.array(
                [
                    [43.1, -32.5],
                    [0.52, 50.35],
                    [40.0, 0.3],
                ]
            ),
            jnp.array([0.50, 0.205, 0.25]),
            0.802,
            533,
        ],
        # AR3, one set of inits and two
        # sets of coefficients
        [
            jnp.array([50.0, 49.9, 48.2]).reshape((3, -1)),
            jnp.array([[0.05, 0.025], [0.25, 0.25], [0.1, 0.1]]),
            0.5,
            1230,
        ],
        # AR3, twos set of (identical) inits, two
        # sets of coefficients, two s.ds
        [
            jnp.array(
                [
                    [50.0, 49.9, 48.2],
                    [50.0, 49.9, 48.2],
                ]
            ).reshape((3, -1)),
            jnp.array([[0.05, 0.025], [0.25, 0.25], [0.1, 0.1]]),
            jnp.array([1, 0.25]),
            1230,
        ],
        # AR3, twos set of (identical) inits, two
        # sets of coefficients, two s.ds,
        # n shorter than the order
        [
            jnp.array(
                [
                    [50.0, 49.9, 48.2],
                    [50.0, 49.9, 48.2],
                ]
            ).reshape((3, -1)),
            jnp.array([[0.05, 0.025], [0.25, 0.25], [0.1, 0.1]]),
            jnp.array([1, 0.25]),
            1,
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
        non_time_dims = jnp.broadcast_shapes(
            jnp.atleast_1d(autoreg).shape[1:],
            jnp.atleast_1d(init_vals).shape[1:],
            jnp.shape(noise_sd),
        )

        expected_shape = (n,) + non_time_dims
        first_entries_broadcast_shape = (order,) + non_time_dims

        expected_first_entries = jnp.broadcast_to(
            init_vals, first_entries_broadcast_shape
        )[:n]

        assert jnp.shape(res) == expected_shape
        assert_array_almost_equal(res[:order, ...], expected_first_entries)


@pytest.mark.parametrize(
    ["init_vals", "autoreg", "noise_sd", "n", "error_match"],
    [
        # autoreg higher dim than init vals
        # and not reshaped appropriately
        [
            jnp.array([50.0, 49.9, 48.2]),
            jnp.array([[0.05, 0.025, 0.025]]),
            0.5,
            1230,
            "Initial values array",
        ],
        # initial vals higher dim than autoreg
        # and not reshaped appropriately
        [
            jnp.array([[50.0, 49.9, 48.2]]),
            jnp.array([0.05, 0.025, 0.025]),
            0.5,
            1230,
            "Initial values array",
        ],
        # not enough initial values
        [
            jnp.array([50.0, 49.9, 48.2]),
            jnp.array([0.05, 0.025, 0.025, 0.25]),
            0.5,
            1230,
            "Initial values array",
        ],
        # too many initial values
        [
            jnp.array([50.0, 49.9, 48.2, 0.035, 0.523]),
            jnp.array([0.05, 0.025, 0.025, 0.25]),
            0.5,
            1230,
            "Initial values array",
        ],
        # unbroadcastable shapes
        [
            jnp.array([[50.0, 49.9], [48.2, 0.035]]),
            jnp.array([[0.05, 0.025], [0.025, 0.25]]),
            jnp.array([0.5, 0.25, 0.3]),
            1230,
            "Incompatible shapes",
        ],
        [
            jnp.array([50.0, 49.9]),
            jnp.array([0.05, 0.025]),
            jnp.array([0.5]),
            1230,
            "Could not broadcast init_vals",
        ],
        # unbroadcastable shapes:
        # sd versus AR mismatch
        [
            jnp.array([50.0, 49.9, 0.25]),
            jnp.array([[0.05, 0.025], [0.025, 0.25], [0.01, 0.1]]),
            jnp.array([0.25, 0.25]),
            1230,
            "Could not broadcast init_vals",
        ],
    ],
)
def test_ar_shape_validation(init_vals, autoreg, noise_sd, n, error_match):
    """
    Test that AR process sample() method validates
    the shapes of its inputs as expected.
    """
    # vector valued noise raises
    # error
    ar = ARProcess()

    # bad dimensionality raises error
    with pytest.raises(ValueError, match=error_match):
        with numpyro.handlers.seed(rng_seed=5):
            ar(
                noise_name="test_ar_noise",
                n=n,
                init_vals=init_vals,
                autoreg=autoreg,
                noise_sd=noise_sd,
            )


@pytest.mark.parametrize(
    ["ar_inits", "autoreg", "noise_sd", "n"],
    [
        [
            jnp.array([25.0]),
            jnp.array([0.75]),
            jnp.array([0.5]),
            10000,
        ],
        [
            jnp.array([-500, -499.0]),
            jnp.array([0.5, 0.45]),
            jnp.array(1.25),
            10001,
        ],
    ],
)
def test_ar_process_asymptotics(ar_inits, autoreg, noise_sd, n):
    """
    Check that AR processes can
    start away from the stationary
    distribution and converge to it.
    """
    ar = ARProcess()
    order = jnp.shape(ar_inits)[0]
    non_time_dims = jnp.broadcast_shapes(
        jnp.atleast_1d(autoreg).shape[1:],
        jnp.atleast_1d(ar_inits).shape[1:],
        jnp.shape(noise_sd),
    )

    first_entries_broadcast_shape = (order,) + non_time_dims

    expected_first_entries = jnp.broadcast_to(
        ar_inits, first_entries_broadcast_shape
    )[:n]

    with numpyro.handlers.seed(rng_seed=62):
        # check it regresses to mean
        # when started away from it
        long_ts = ar(
            noise_name="arprocess_noise",
            n=n,
            init_vals=ar_inits,
            autoreg=autoreg,
            noise_sd=noise_sd,
        )
        assert_array_almost_equal(long_ts[:order], expected_first_entries)

        assert jnp.abs(long_ts[-1]) < 3 * noise_sd
