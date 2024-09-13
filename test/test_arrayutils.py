"""
Tests for the arrayutils module.
"""

import jax.numpy as jnp
import pytest

import pyrenew.arrayutils as au


def test_pad_edges_to_match():
    """
    Test function to verify padding along the edges for 1D and 2D arrays
    """

    # test when y gets padded
    x = jnp.array([1, 2, 3])
    y = jnp.array([1, 2])

    x_pad, y_pad = au.pad_edges_to_match(x, y)
    assert x_pad.size == y_pad.size
    assert y_pad[-1] == y[-1]
    assert jnp.array_equal(x_pad, x)

    # test when x gets padded
    x = jnp.array([1, 2])
    y = jnp.array([1, 2, 3])

    x_pad, y_pad = au.pad_edges_to_match(x, y)
    assert x_pad.size == y_pad.size
    assert x_pad[-1] == x[-1]
    assert jnp.array_equal(y_pad, y)

    # test when no padding required
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    x_pad, y_pad = au.pad_edges_to_match(x, y)

    assert jnp.array_equal(x_pad, x)
    assert jnp.array_equal(y_pad, y)

    # Verify function works with both padding directions
    x = jnp.array([1, 2, 3])
    y = jnp.array([1, 2])

    x_pad, y_pad = au.pad_edges_to_match(x, y, pad_direction="start")

    assert x_pad.size == y_pad.size
    assert y_pad[0] == y[0]
    assert jnp.array_equal(x_pad, x)

    # Verify that the function raises an error when `fix_y` is True
    with pytest.raises(
        ValueError, match="Cannot fix y when x is longer than y"
    ):
        x_pad, y_pad = au.pad_edges_to_match(x, y, fix_y=True)

    # Verify function raises an error when pad_direction is not "start" or "end"
    with pytest.raises(ValueError):
        x_pad, y_pad = au.pad_edges_to_match(x, y, pad_direction="middle")

    # test padding for 2D arrays
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[5, 6]])

    # Padding along axis 0
    axis = 0
    x_pad, y_pad = au.pad_edges_to_match(x, y, axis=axis, pad_direction="end")

    assert jnp.array_equal(x_pad.shape[axis], y_pad.shape[axis])
    assert jnp.array_equal(y_pad[-1], y[-1])
    assert jnp.array_equal(x_pad, x)

    # padding along axis 1
    axis = 1
    x_pad, y_pad = au.pad_edges_to_match(x, y, axis=axis, pad_direction="end")
    assert jnp.array_equal(x_pad.shape[axis], y_pad.shape[axis])
    assert jnp.array_equal(y[:, -1], y_pad[:, -1])
    assert jnp.array_equal(x_pad, x)
