"""
Tests for the arrayutils module.
"""

import jax.numpy as jnp
import pyrenew.arrayutils as au
import pytest


def test_arrayutils_pad_to_match():
    """
    Verifies extension when required and error when `fix_y` is True.
    """

    x = jnp.array([1, 2, 3])
    y = jnp.array([1, 2])

    x_pad, y_pad = au.pad_to_match(x, y)

    assert x_pad.size == y_pad.size
    assert x_pad.size == 3

    x = jnp.array([1, 2])
    y = jnp.array([1, 2, 3])

    x_pad, y_pad = au.pad_to_match(x, y)

    assert x_pad.size == y_pad.size
    assert x_pad.size == 3

    x = jnp.array([1, 2, 3])
    y = jnp.array([1, 2])

    # Verify that the function raises an error when `fix_y` is True
    with pytest.raises(ValueError):
        x_pad, y_pad = au.pad_to_match(x, y, fix_y=True)


def test_arrayutils_pad_x_to_match_y():
    """
    Verifies extension when required
    """

    x = jnp.array([1, 2])
    y = jnp.array([1, 2, 3])

    x_pad = au.pad_x_to_match_y(x, y)

    assert x_pad.size == 3
