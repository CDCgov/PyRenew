"""
Tests for the arrayutils module.
"""

import jax.numpy as jnp
import numpy as np
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

    # Verify function works with both padding directions
    x_pad, y_pad = au.pad_to_match(x, y, pad_direction="start")

    assert x_pad.size == y_pad.size
    assert x_pad.size == 3

    # Verify function raises an error when pad_direction is not "start" or "end"
    with pytest.raises(ValueError):
        x_pad, y_pad = au.pad_to_match(x, y, pad_direction="middle")


def test_arrayutils_pad_x_to_match_y():
    """
    Verifies extension when required
    """

    x = jnp.array([1, 2])
    y = jnp.array([1, 2, 3])

    x_pad = au.pad_x_to_match_y(x, y)

    assert x_pad.size == 3


def test_validate_arraylike():
    """
    Test the validate_arraylike
    validation function provided
    in arrayutils raises an informative
    error when non-ArrayLike objects
    are passed but does not raise an error
    when ArrayLike objects are passed.
    """
    for non_array in [
            [], {}, "a_string",
            lambda x: x + 5]:
        with pytest.raises(ValueError):
            au.validate_arraylike(non_array,
                                  "a_test_non_array")

    for my_arraylike in [
            352.0,
            5,
            jnp.array([3, 2, 3]),
            np.array([1, 2, 3]),
            jnp.array(5.344),
            np.array(-32.32),
            jnp.array([[[3, 2.3, 1]]]),
            np.array([[[3, 2.3, 1]]])
    ]:
        au.validate_arraylike(my_arraylike, "a_test_arraylike")
