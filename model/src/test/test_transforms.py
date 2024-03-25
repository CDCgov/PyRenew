# -*- coding: utf-8 -*-

"""
Tests for transformations
"""

import jax.numpy as jnp
import pyrenew.transform as t
from numpy.testing import assert_array_almost_equal


def generic_inversion_test(transform, test_vals, decimal=1e-8, **kwargs):
    """
    Generic test for inverting a
    pyrenew transform, confirming
    that f^-1(f(x)) = x for the
    x values givein in `test_vals`

    Parameters
    -----------
    transform : pyrenew.transform.AbstractTransform
        Uninstantiated transformation to instantiate
        and test

    test_vals : ArrayLike
        Array of test values on which to test
        applying and then inverting the transform

    decimal : float
        Decimal tolerance, passed to
        numpy.testing.assert_array_almost_equal()

    **kwargs :
        Additional keyword arguments passed
        to the transform constructor
    """
    instantiated = transform(**kwargs)

    assert_array_almost_equal(
        test_vals,
        instantiated.inverse(instantiated(test_vals)),
        decimal=decimal,
    )
    assert_array_almost_equal(
        test_vals,
        instantiated.inverse(instantiated.transform(test_vals)),
        decimal=decimal,
    )


def test_invert_dists():
    generic_inversion_test(
        t.LogTransform, jnp.array([1.52, 0.21, 1563.52, 23.523, 1.2352e7])
    )
    generic_inversion_test(
        t.LogitTransform, jnp.array([0.99235, 0.13242, 0.5, 0.235, 0.862])
    )
    generic_inversion_test(
        t.ScaledLogitTransform,
        50 * jnp.array([0.99235, 0.13242, 0.5, 0.235, 0.862]),
        x_max=50,
    )
    generic_inversion_test(
        t.IdentityTransform, jnp.array([0.99235, 0.13242, 0.5, 0.235, 0.862])
    )
