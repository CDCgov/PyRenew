"""
Unit tests for the iterative convolution
scanner function factories found in pyrenew.convolve
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import pyrenew.convolve as pc
import pyrenew.transformation as t


@pytest.mark.parametrize(
    ["inits", "to_scan_a", "multipliers"],
    [
        [
            jnp.array([0.352, 5.2, -3]),
            jnp.array([0.5, 0.3, 0.2]),
            jnp.array(np.random.normal(0, 0.5, size=500)),
        ],
        [
            jnp.array(np.array([0.352, 5.2, -3] * 3).reshape(3, 3)),
            jnp.array([0.5, 0.3, 0.2]),
            jnp.array(np.random.normal(0, 0.5, size=(500, 3))),
        ],
    ],
)
def test_double_scanner_reduces_to_single(inits, to_scan_a, multipliers):
    """
    Test that new_double_scanner() yields a function
    that is equivalent to a single scanner if the first
    scan is chosen appropriately
    """

    def transform_a(x: any):
        """
        transformation associated with
        array to_scan_a

        Parameters
        ----------
        x: any
            input value

        Returns
        -------
        The result of 4 * x + 0.025, where x is the input
        value
        """
        return 4 * x + 0.025

    scanner_a = pc.new_convolve_scanner(to_scan_a, transform_a)

    double_scanner_a = pc.new_double_convolve_scanner(
        (jnp.array([523, 2, -0.5233]), to_scan_a),
        (jnp.ones_like, transform_a),
    )

    _, result_a = jax.lax.scan(f=scanner_a, init=inits, xs=multipliers)

    _, result_a_double = jax.lax.scan(
        f=double_scanner_a, init=inits, xs=(multipliers * 0.2352, multipliers)
    )

    assert_array_equal(result_a_double[1], jnp.ones_like(multipliers))
    assert_array_equal(result_a_double[0], result_a)


@pytest.mark.parametrize(
    ["arr", "history", "multipliers", "transform"],
    [
        [
            jnp.array([1.0, 2.0]),
            jnp.array([3.0, 4.0]),
            jnp.array([1, 2, 3]),
            t.IdentityTransform(),
        ],
        [
            jnp.ones(3),
            jnp.array(np.array([0.5, 0.3, 0.2] * 3)).reshape(3, 3),
            jnp.ones((3, 3)),
            t.ExpTransform(),
        ],
    ],
)
def test_convolve_scanner_using_scan(arr, history, multipliers, transform):
    """
    Tests the output of new convolve scanner function
    used with `jax.lax.scan` against values calculated
    using a for loop
    """
    scanner = pc.new_convolve_scanner(arr, transform)

    _, result = jax.lax.scan(f=scanner, init=history, xs=multipliers)

    result_not_scanned = []
    for multiplier in multipliers:
        history, new_val = scanner(history, multiplier)
        result_not_scanned.append(new_val)

    assert jnp.array_equal(result, result_not_scanned)


@pytest.mark.parametrize(
    ["arr1", "arr2", "history", "m1", "m2", "transform"],
    [
        [
            jnp.array([1.0, 2.0]),
            jnp.array([2.0, 1.0]),
            jnp.array([0.1, 0.4]),
            jnp.array([1, 2, 3]),
            jnp.ones(3),
            (t.IdentityTransform(), t.IdentityTransform()),
        ],
        [
            jnp.array([1.0, 2.0, 0.3]),
            jnp.array([2.0, 1.0, 0.5]),
            jnp.array(np.array([0.5, 0.3, 0.2] * 3)).reshape(3, 3),
            jnp.ones((3, 3)),
            jnp.ones((3, 3)),
            (t.ExpTransform(), t.IdentityTransform()),
        ],
    ],
)
def test_double_convolve_scanner_using_scan(
    arr1, arr2, history, m1, m2, transform
):
    """
    Tests the output of new convolve double scanner function
    used with `jax.lax.scan` against values calculated
    using a for loop
    """
    arr1 = jnp.array([1.0, 2.0])
    arr2 = jnp.array([2.0, 1.0])
    transform = (t.IdentityTransform(), t.IdentityTransform())
    history = jnp.array([0.1, 0.4])
    m1, m2 = (jnp.array([1, 2, 3]), jnp.ones(3))

    scanner = pc.new_double_convolve_scanner((arr1, arr2), transform)

    _, result = jax.lax.scan(f=scanner, init=history, xs=(m1, m2))

    res1, res2 = [], []
    for m1, m2 in zip(m1, m2):
        history, new_val = scanner(history, (m1, m2))
        res1.append(new_val[0])
        res2.append(new_val[1])

    assert jnp.array_equal(result, (res1, res2))


@pytest.mark.parametrize(
    ["arr", "history", "multiplier", "transform"],
    [
        [
            jnp.array([1.0, 2.0]),
            jnp.array([3.0, 4.0]),
            jnp.array(2),
            t.IdentityTransform(),
        ],
        [
            jnp.ones(3),
            jnp.array(np.array([0.5, 0.3, 0.2] * 3)).reshape(3, 3),
            jnp.ones(3),
            t.ExpTransform(),
        ],
    ],
)
def test_convolve_scanner(arr, history, multiplier, transform):
    """
    Tests new convolve scanner function
    """
    scanner = pc.new_convolve_scanner(arr, transform)
    latest, new_val = scanner(history, multiplier)
    assert jnp.array_equal(
        new_val, transform(multiplier * jnp.dot(arr, history))
    )


@pytest.mark.parametrize(
    ["arr1", "arr2", "history", "m1", "m2", "transforms"],
    [
        [
            jnp.array([1.0, 2.0]),
            jnp.array([2.0, 1.0]),
            jnp.array([0.1, 0.4]),
            jnp.array(1),
            jnp.array(3),
            (t.IdentityTransform(), t.IdentityTransform()),
        ],
        [
            jnp.array([1.0, 2.0, 0.3]),
            jnp.array([2.0, 1.0, 0.5]),
            jnp.array(np.array([0.5, 0.3, 0.2] * 3)).reshape(3, 3),
            jnp.ones(3),
            0.1 * jnp.ones(3),
            (t.ExpTransform(), t.IdentityTransform()),
        ],
    ],
)
def test_double_convolve_scanner(arr1, arr2, history, m1, m2, transforms):
    """
    Tests new double convolve scanner function
    """
    double_scanner = pc.new_double_convolve_scanner((arr1, arr2), transforms)
    latest, (new_val, m_net) = double_scanner(history, (m1, m2))

    assert jnp.array_equal(m_net, transforms[0](m1 * jnp.dot(arr1, history)))
    assert jnp.array_equal(
        new_val, transforms[1](m2 * m_net * jnp.dot(arr2, history))
    )
