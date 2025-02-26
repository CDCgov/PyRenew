"""
Tests for the pyrenew.time module.
"""

import jax.numpy as jnp
import pytest

import pyrenew.time as ptime


def test_validate_dow() -> None:
    """
    Test that validate_dow raises appropriate
    errors and succeeds on valid values.
    """
    with pytest.raises(ValueError, match="which is a"):
        ptime.validate_dow("a string", "var")
    with pytest.raises(ValueError, match="Got -1 for var"):
        ptime.validate_dow(-1, "var")
    with pytest.raises(ValueError, match="Got 7 for other_var"):
        ptime.validate_dow(7, "other_var")
    assert [ptime.validate_dow(x, "valid") for x in range(7)] == [None] * 7


def test_daily_to_weekly_no_offset():
    """
    Tests that the function correctly aggregates
    daily values into weekly totals when there
    is no offset both input and output start dow on Monday.
    """
    daily_values = jnp.arange(1, 15)
    result = ptime.daily_to_weekly(daily_values)
    expected = jnp.array([28, 77])
    assert jnp.array_equal(result, expected)


def test_daily_to_weekly_with_input_data_offset():
    """
    Tests that the function correctly aggregates
    daily values into weekly totals with dow
    offset in the input data.
    """
    daily_values = jnp.arange(1, 15)
    result = ptime.daily_to_weekly(daily_values, input_data_first_dow=2)
    expected = jnp.array([63])
    assert jnp.array_equal(result, expected)


def test_daily_to_weekly_with_different_week_start():
    """
    Tests aggregation when the desired week start
    differs from the input data start.
    """
    daily_values = jnp.arange(1, 15)
    result = ptime.daily_to_weekly(
        daily_values, input_data_first_dow=2, week_start_dow=5
    )
    expected = jnp.array([49])
    assert jnp.array_equal(result, expected)


def test_daily_to_weekly_incomplete_week():
    """
    Tests that the function raises a
    ValueError when there are
    insufficient daily values to
    form a complete week.
    """
    daily_values = jnp.arange(1, 5)
    with pytest.raises(
        ValueError, match="No complete weekly values available"
    ):
        ptime.daily_to_weekly(daily_values, input_data_first_dow=0)


def test_daily_to_weekly_missing_daily_values():
    """
    Tests that the function correctly
    aggregates the available daily values
    into weekly values when there are
    fewer daily values than required for
    complete weekly totals in the final week.
    """
    daily_values = jnp.arange(1, 10)
    result = ptime.daily_to_weekly(daily_values, input_data_first_dow=0)
    expected = jnp.array([28])
    assert jnp.array_equal(result, expected)


def test_daily_to_weekly_invalid_offset():
    """
    Tests that the function raises a
    ValueError when the offset is
    outside the valid range (0-6).
    """
    daily_values = jnp.arange(1, 15)
    with pytest.raises(
        ValueError,
        match="Got -1 for input_data_first_dow",
    ):
        ptime.daily_to_weekly(daily_values, input_data_first_dow=-1)

    with pytest.raises(
        ValueError,
        match="Got 7 for week_start_dow",
    ):
        ptime.daily_to_weekly(daily_values, week_start_dow=7)


def test_daily_to_mmwr_epiweekly():
    """
    Tests aggregation for MMWR epidemiological week.
    """
    daily_values = jnp.arange(1, 15)
    result = ptime.daily_to_mmwr_epiweekly(daily_values)
    expected = jnp.array([70])
    assert jnp.array_equal(result, expected)
