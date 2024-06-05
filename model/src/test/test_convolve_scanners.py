import pyrenew.convolve as pc
import jax.numpy as jnp
import numpy as np
import jax
from numpy.testing import assert_array_equal


def test_double_scanner_reduces_to_single():
    """
    Test that new_double_scanner() yields a function
    that is equivalent to a single scanner if the first
    scan is chosen appropriately
    """
    inits = jnp.array([0.352, 5.2, -3])
    to_scan_a = jnp.array([0.5, 0.3, 0.2])

    multipliers = jnp.array(
        np.random.normal(0, 0.5, size=500))

    def transform_a(x):
        return x / 0.25 + 0.025

    scanner_a = pc.new_convolve_scanner(to_scan_a, transform_a)

    double_scanner_a = pc.new_double_scanner(
        (jnp.array([523, 2, -0.5233]),
         to_scan_a),
        (lambda x: 1, transform_a))

    _, result_a = jax.lax.scan(
        f=scanner_a,
        init=inits,
        xs=multipliers)

    _, result_a_double = jax.lax.scan(
        f=double_scanner_a,
        init=inits,
        xs=(multipliers * 0.2352, multipliers))

    assert_array_equal(
        result_a_double[1], jnp.ones_like(multipliers))
    assert_array_equal(
        result_a_double[0], result_a)
