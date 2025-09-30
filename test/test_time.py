"""
Tests for the pyrenew.time module.
"""

import datetime as dt
import itertools

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_array_equal

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
    with pytest.raises(ValueError, match="No complete weekly values available"):
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
    expected = jnp.array([28, 77])
    assert jnp.array_equal(result, expected)
    # for any starting day of the week in the input data
    # except Sunday (6), the result should throw out partial
    # first and last weeks
    for dow in range(6):
        first_sunday_ind = (6 - dow) % 7
        minus_last_sunday_ind = (dow - 6) % 7
        expected_partial = 1.0 * jnp.atleast_1d(
            jnp.sum(daily_values[first_sunday_ind:-minus_last_sunday_ind])
        )
        result_partial = ptime.daily_to_mmwr_epiweekly(
            daily_values, input_data_first_dow=dow
        )
        assert jnp.array_equal(result_partial, expected_partial)


@pytest.mark.parametrize(
    "weekly_values, expected_output",
    [
        (
            jnp.array([7, -8, 501]),
            jnp.array(
                [
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    -8,
                    -8,
                    -8,
                    -8,
                    -8,
                    -8,
                    -8,
                    501,
                    501,
                    501,
                    501,
                    501,
                    501,
                    501,
                ]
            ),
        ),
        (
            jnp.array([[7, 80], [14, 15]]),
            jnp.array(
                [
                    [7, 80],
                    [7, 80],
                    [7, 80],
                    [7, 80],
                    [7, 80],
                    [7, 80],
                    [7, 80],
                    [14, 15],
                    [14, 15],
                    [14, 15],
                    [14, 15],
                    [14, 15],
                    [14, 15],
                    [14, 15],
                ]
            ),
        ),
    ],
)
def test_weekly_to_daily_output(weekly_values, expected_output):
    """
    Test that weekly_to_daily works as expected with full
    and with partial weeks, and that mmwr_epiweekly_to_daily
    is equivalent to weekly_to_daily with week_start_dow=6.
    """
    assert_array_equal(ptime.weekly_to_daily(weekly_values), expected_output)
    for out_first_dow, input_start_dow in itertools.product(range(7), range(7)):
        expected_offset = (out_first_dow - input_start_dow) % 7
        assert_array_equal(
            ptime.weekly_to_daily(
                weekly_values,
                output_data_first_dow=out_first_dow,
                week_start_dow=input_start_dow,
            ),
            expected_output[expected_offset:],
        )
        if input_start_dow == 6:
            assert_array_equal(
                ptime.mmwr_epiweekly_to_daily(
                    weekly_values, output_data_first_dow=out_first_dow
                ),
                expected_output[expected_offset:],
            )


def test_daily_to_weekly_2d_values_with_offset():
    """
    Tests that the function correctly aggregates
    2D daily values into weekly totals with dow
    offset in the input data.
    """
    daily_values_to_broadcast = jnp.arange(1, 4)
    desired_weeks = 3
    provided_days = desired_weeks * 7 + 4
    t_zeros = 2

    daily_values = jnp.broadcast_to(
        daily_values_to_broadcast,
        (provided_days, daily_values_to_broadcast.size),
    )

    daily_values_with_leading_zeros = jnp.concatenate(
        [jnp.zeros((t_zeros, daily_values_to_broadcast.size)), daily_values],
        axis=0,
    )

    result = ptime.daily_to_weekly(daily_values)
    result_leading_zero_no_offeset = ptime.daily_to_weekly(
        daily_values_with_leading_zeros
    )
    result_leading_zero_offeset = ptime.daily_to_weekly(
        daily_values_with_leading_zeros, input_data_first_dow=t_zeros
    )
    expected = jnp.repeat(
        7 * daily_values_to_broadcast[jnp.newaxis, :], desired_weeks, axis=0
    )
    expected_leading_zero = jnp.array([[5, 10, 15], [7, 14, 21], [7, 14, 21]])

    assert jnp.array_equal(result, expected)
    assert jnp.array_equal(result_leading_zero_offeset, expected)
    assert jnp.array_equal(result_leading_zero_no_offeset, expected_leading_zero)


# validate_mmwr_dates tests
def test_validate_mmwr_dates_valid_saturdays():
    """Valid Saturday dates should not raise."""
    saturdays = [
        dt.datetime(2025, 1, 4),  # Saturday
        dt.datetime(2025, 1, 11),  # Saturday
        np.datetime64("2025-01-18"),  # Saturday
    ]
    ptime.validate_mmwr_dates(saturdays)  # Should not raise


def test_validate_mmwr_dates_invalid_weekday():
    """Non-Saturday dates should raise ValueError."""
    with pytest.raises(ValueError, match="MMWR dates must be Saturdays"):
        ptime.validate_mmwr_dates([dt.datetime(2025, 1, 6)])  # Monday


def test_validate_mmwr_dates_with_none():
    """None values should be skipped."""
    dates_with_none = [
        None,
        dt.datetime(2025, 1, 4),  # Saturday
        None,
        np.datetime64("2025-01-11"),  # Saturday
    ]
    ptime.validate_mmwr_dates(dates_with_none)  # Should not raise


def test_validate_mmwr_dates_mixed_types():
    """Mix of datetime and np.datetime64 should work."""
    mixed_dates = [
        dt.datetime(2025, 1, 4),  # Saturday
        np.datetime64("2025-01-11"),  # Saturday
    ]
    ptime.validate_mmwr_dates(mixed_dates)  # Should not raise


def test_validate_mmwr_dates_empty_array():
    """Empty array should not raise."""
    ptime.validate_mmwr_dates([])  # Should not raise


# date_to_model_t tests
def test_date_to_model_t_same_date():
    """Same date as start_date should return 0."""
    start = dt.datetime(2025, 1, 1)
    assert ptime.date_to_model_t(start, start) == 0


def test_date_to_model_t_future_date():
    """Future dates should return positive integers."""
    start = dt.datetime(2025, 1, 1)
    future = dt.datetime(2025, 1, 15)
    assert ptime.date_to_model_t(future, start) == 14


def test_date_to_model_t_past_date():
    """Past dates should return negative integers."""
    start = dt.datetime(2025, 1, 15)
    past = dt.datetime(2025, 1, 1)
    assert ptime.date_to_model_t(past, start) == -14


def test_date_to_model_t_datetime_types():
    """Test all combinations of datetime and np.datetime64."""
    start_dt = dt.datetime(2025, 1, 1)
    start_np = np.datetime64("2025-01-01")
    date_dt = dt.datetime(2025, 1, 15)
    date_np = np.datetime64("2025-01-15")

    assert ptime.date_to_model_t(date_dt, start_dt) == 14
    assert ptime.date_to_model_t(date_dt, start_np) == 14
    assert ptime.date_to_model_t(date_np, start_dt) == 14
    assert ptime.date_to_model_t(date_np, start_np) == 14


def test_date_to_model_t_leap_year():
    """Verify correct calculation across Feb 29."""
    start = dt.datetime(2024, 2, 28)
    after_leap = dt.datetime(2024, 3, 1)
    assert ptime.date_to_model_t(after_leap, start) == 2


def test_date_to_model_t_year_boundary():
    """Test calculation across year boundary."""
    start = dt.datetime(2024, 12, 30)
    new_year = dt.datetime(2025, 1, 2)
    assert ptime.date_to_model_t(new_year, start) == 3


# model_t_to_date tests
def test_model_t_to_date_t_zero():
    """t=0 should return start_date."""
    start = dt.datetime(2025, 1, 1)
    result = ptime.model_t_to_date(0, start)
    assert result == start


def test_model_t_to_date_positive_t():
    """Positive t should return future dates."""
    start = dt.datetime(2025, 1, 1)
    result = ptime.model_t_to_date(14, start)
    assert result == dt.datetime(2025, 1, 15)


def test_model_t_to_date_negative_t():
    """Negative t should return past dates."""
    start = dt.datetime(2025, 1, 15)
    result = ptime.model_t_to_date(-14, start)
    assert result == dt.datetime(2025, 1, 1)


def test_model_t_to_date_roundtrip():
    """Verify roundtrip consistency with date_to_model_t."""
    start = dt.datetime(2025, 1, 1)
    for t in [-10, 0, 7, 30, 365]:
        date = ptime.model_t_to_date(t, start)
        assert ptime.date_to_model_t(date, start) == t


def test_model_t_to_date_input_types():
    """Test datetime vs np.datetime64 start_date."""
    start_dt = dt.datetime(2025, 1, 1)
    start_np = np.datetime64("2025-01-01")

    result_dt = ptime.model_t_to_date(14, start_dt)
    result_np = ptime.model_t_to_date(14, start_np)

    assert result_dt == dt.datetime(2025, 1, 15)
    assert result_np == dt.datetime(2025, 1, 15)


# get_date_range_length tests
def test_get_date_range_length_default_timestep():
    """Test with default timestep_days=1."""
    dates = np.array(
        [
            np.datetime64("2025-01-01"),
            np.datetime64("2025-01-15"),
        ]
    )
    assert ptime.get_date_range_length(dates) == 15


def test_get_date_range_length_weekly_timestep():
    """Test with timestep_days=7."""
    dates = np.array(
        [
            np.datetime64("2025-01-01"),
            np.datetime64("2025-01-29"),
        ]
    )
    assert ptime.get_date_range_length(dates, timestep_days=7) == 5


def test_get_date_range_length_single_date():
    """Single date should return 1."""
    dates = np.array([np.datetime64("2025-01-01")])
    assert ptime.get_date_range_length(dates) == 1


def test_get_date_range_length_multiple_dates():
    """Test with multiple dates in array."""
    dates = np.array(
        [
            np.datetime64("2025-01-01"),
            np.datetime64("2025-01-08"),
            np.datetime64("2025-01-15"),
            np.datetime64("2025-01-31"),
        ]
    )
    # Should use min to max
    assert ptime.get_date_range_length(dates) == 31


# get_end_date tests
def test_get_end_date_basic():
    """Test basic calculation with various n_points."""
    start = dt.datetime(2025, 1, 1)
    assert ptime.get_end_date(start, 1) == np.datetime64("2025-01-01")
    assert ptime.get_end_date(start, 7) == np.datetime64("2025-01-07")
    assert ptime.get_end_date(start, 31) == np.datetime64("2025-01-31")


def test_get_end_date_n_points_one():
    """n_points=1 should return start_date."""
    start = dt.datetime(2025, 1, 15)
    result = ptime.get_end_date(start, 1)
    assert result == np.datetime64("2025-01-15")


def test_get_end_date_negative_n_points():
    """Negative n_points should raise ValueError."""
    start = dt.datetime(2025, 1, 1)
    with pytest.raises(ValueError, match="n_points must be non-negative"):
        ptime.get_end_date(start, -5)


def test_get_end_date_none_start_with_zero_points():
    """start_date=None with n_points=0 should return None."""
    result = ptime.get_end_date(None, 0)
    assert result is None


def test_get_end_date_none_start_with_positive_points():
    """start_date=None with n_points>0 should raise ValueError."""
    with pytest.raises(ValueError, match="Must provide start_date"):
        ptime.get_end_date(None, 5)


def test_get_end_date_weekly_timestep():
    """Test with timestep_days=7."""
    start = dt.datetime(2025, 1, 1)
    result = ptime.get_end_date(start, 4, timestep_days=7)
    assert result == np.datetime64("2025-01-22")


def test_get_end_date_input_types():
    """Test datetime vs np.datetime64 input."""
    start_dt = dt.datetime(2025, 1, 1)
    start_np = np.datetime64("2025-01-01")

    result_dt = ptime.get_end_date(start_dt, 10)
    result_np = ptime.get_end_date(start_np, 10)

    assert result_dt == np.datetime64("2025-01-10")
    assert result_np == np.datetime64("2025-01-10")


# get_n_data_days tests
def test_get_n_data_days_with_n_points():
    """Test with only n_points specified."""
    assert ptime.get_n_data_days(n_points=15) == 15
    assert ptime.get_n_data_days(n_points=0) == 0


def test_get_n_data_days_with_date_array():
    """Test with only date_array specified."""
    dates = np.array(
        [
            np.datetime64("2025-01-01"),
            np.datetime64("2025-01-15"),
        ]
    )
    assert ptime.get_n_data_days(date_array=dates) == 15


def test_get_n_data_days_neither():
    """Both None should return 0."""
    assert ptime.get_n_data_days() == 0


def test_get_n_data_days_both():
    """Both specified should raise ValueError."""
    dates = np.array([np.datetime64("2025-01-01")])
    with pytest.raises(ValueError, match="Must provide at most one"):
        ptime.get_n_data_days(n_points=10, date_array=dates)


def test_get_n_data_days_weekly_timestep():
    """Test timestep_days=7 with date_array."""
    dates = np.array(
        [
            np.datetime64("2025-01-01"),
            np.datetime64("2025-01-29"),
        ]
    )
    assert ptime.get_n_data_days(date_array=dates, timestep_days=7) == 5


def test_create_date_time_spine_daily():
    """Test with daily frequency."""
    start = dt.datetime(2025, 1, 1)
    end = dt.datetime(2025, 1, 5)
    result = ptime.create_date_time_spine(start, end, freq="1d")

    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 5
    assert "date" in result.columns
    assert "t" in result.columns
    assert len(result.columns) == 2


def test_create_date_time_spine_columns():
    """Verify output has correct columns."""
    start = dt.datetime(2025, 1, 1)
    end = dt.datetime(2025, 1, 3)
    result = ptime.create_date_time_spine(start, end)

    assert "date" in result.columns
    assert "t" in result.columns
    assert result.schema["t"] == pl.Int64


def test_create_date_time_spine_t_starts_at_zero():
    """First row should have t=0."""
    start = dt.datetime(2025, 1, 1)
    end = dt.datetime(2025, 1, 5)
    result = ptime.create_date_time_spine(start, end)

    assert result["t"][0] == 0


def test_create_date_time_spine_t_increments():
    """t should increment correctly."""
    start = dt.datetime(2025, 1, 1)
    end = dt.datetime(2025, 1, 5)
    result = ptime.create_date_time_spine(start, end)

    assert list(result["t"]) == [0, 1, 2, 3, 4]


def test_create_date_time_spine_single_day():
    """Test with start_date == end_date."""
    start = dt.datetime(2025, 1, 15)
    end = dt.datetime(2025, 1, 15)
    result = ptime.create_date_time_spine(start, end)

    assert result.shape[0] == 1
    assert result["t"][0] == 0


def test_create_date_time_spine_input_types():
    """Test different input date types."""
    start_dt = dt.datetime(2025, 1, 1)
    end_dt = dt.datetime(2025, 1, 3)
    start_np = np.datetime64("2025-01-01")
    end_np = np.datetime64("2025-01-03")

    result_dt = ptime.create_date_time_spine(start_dt, end_dt)
    result_np = ptime.create_date_time_spine(start_np, end_np)
    result_mixed = ptime.create_date_time_spine(start_dt, end_np)

    assert result_dt.shape[0] == 3
    assert result_np.shape[0] == 3
    assert result_mixed.shape[0] == 3


def test_create_date_time_spine_date_values():
    """Verify date column contains correct dates."""
    start = dt.datetime(2025, 1, 1)
    end = dt.datetime(2025, 1, 3)
    result = ptime.create_date_time_spine(start, end)

    dates = result["date"].to_list()
    assert dates[0] == dt.date(2025, 1, 1)
    assert dates[1] == dt.date(2025, 1, 2)
    assert dates[2] == dt.date(2025, 1, 3)
