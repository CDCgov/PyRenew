"""
Tests for transformations
"""

import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal

import pyrenew.transformation as t


def generic_inversion_test(transform, test_vals, decimal=1e-8, **kwargs) -> None:
    """
    Generic test for inverting a
    pyrenew transform, confirming
    that f^-1(f(x)) = x for the
    x values give in in `test_vals`

    Parameters
    ----------
    transform
        Uninstantiated transformation to instantiate
        and test

    test_vals
        Array of test values on which to test
        applying and then inverting the transform

    decimal
        Decimal tolerance, passed to
        numpy.testing.assert_array_almost_equal()

    **kwargs
        Additional keyword arguments passed
        to the transform constructor
    """
    instantiated = transform(**kwargs)

    assert_array_almost_equal(
        test_vals,
        instantiated.inv(instantiated(test_vals)),
        decimal=decimal,
    )

    return None


def test_invert_dists() -> None:
    """
    Test the inversion of the
    built-in transformations
    """
    generic_inversion_test(
        t.ScaledLogitTransform,
        50 * jnp.array([0.99235, 0.13242, 0.5, 0.235, 0.862]),
        x_max=50,
    )

    return None
