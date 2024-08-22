"""
Unit tests for the iterative convolution
scanner function factories found in pyrenew.convolve
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_equal

import pyrenew.convolve as pc


def test_double_scanner_reduces_to_single():
    """
    Test that new_double_scanner() yields a function
    that is equivalent to a single scanner if the first
    scan is chosen appropriately
    """
    inits = jnp.array([0.352, 5.2, -3])
    to_scan_a = jnp.array([0.5, 0.3, 0.2])

    multipliers = jnp.array(np.random.normal(0, 0.5, size=500))

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
        (jnp.array([523, 2, -0.5233]), to_scan_a), (lambda x: 1, transform_a)
    )

    _, result_a = jax.lax.scan(f=scanner_a, init=inits, xs=multipliers)

    _, result_a_double = jax.lax.scan(
        f=double_scanner_a, init=inits, xs=(multipliers * 0.2352, multipliers)
    )

    assert_array_equal(result_a_double[1], jnp.ones_like(multipliers))
    assert_array_equal(result_a_double[0], result_a)
