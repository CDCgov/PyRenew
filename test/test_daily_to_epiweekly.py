# numpydoc ignore=GL08

import jax.numpy as jnp
import pytest

from pyrenew.convolve import daily_to_epiweekly


def test_daily_to_epiweekly_no_offset():
    """
    Tests that the function correctly aggregates
    daily values into epiweekly totals when there
    is no offset, starting from the first day.
    """
    daily_values = jnp.arange(1, 15)
    result = daily_to_epiweekly(daily_values)
    expected = jnp.array([28, 77])
    assert jnp.array_equal(result, expected)


def test_daily_to_epiweekly_with_offset():
    """
    Tests that the function correctly aggregates
    daily values into epiweekly totals when there
    is a dow offset.
    """
    daily_values = jnp.arange(1, 15)
    result = daily_to_epiweekly(daily_values, first_dow=2)
    expected = jnp.array([63])
    assert jnp.array_equal(result, expected)


def test_daily_to_epiweekly_incomplete_epiweek():
    """
    Tests that the function raises a
    ValueError when there are
    insufficient daily values to
    form a complete epiweek.
    """
    daily_values = jnp.arange(1, 5)
    with pytest.raises(
        ValueError, match="No complete epiweekly values available"
    ):
        daily_to_epiweekly(daily_values, first_dow=0)


def test_daily_to_epiweekly_missing_daily_values():
    """
    Tests that the function correctly
    aggregates the available daily values
    into epiweekly values when there are
    fewer daily values than required for
    complete epiweekly totals.
    """
    daily_values = jnp.arange(1, 10)
    result = daily_to_epiweekly(daily_values, first_dow=0)
    expected = jnp.array([28])
    assert jnp.array_equal(result, expected)


def test_daily_to_epiweekly_invalid_offset():
    """
    Tests that the function raises a
    ValueError when the offset is
    outside the valid range (0-6).
    """
    daily_values = jnp.arange(1, 15)
    with pytest.raises(
        ValueError, match="First day of the week must be between 0 and 6"
    ):
        daily_to_epiweekly(daily_values, first_dow=-1)

    with pytest.raises(
        ValueError, match="First day of the week must be between 0 and 6"
    ):
        daily_to_epiweekly(daily_values, first_dow=7)
