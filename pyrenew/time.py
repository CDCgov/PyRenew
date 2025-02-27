"""
Helper functions for handling timeseries in Pyrenew

Days of the week in pyrenew are 0-indexed and follow
ISO standards, so 0 is Monday at 6 is Sunday.
"""

import jax.numpy as jnp
from jax.typing import ArrayLike


def validate_dow(day_of_week: int, variable_name: str) -> None:
    """
    Confirm that an integer is a valid Pyrenew day of the week
    index, with informative error messages on failure.

    Parameters
    ----------
    day_of_week: int
       Integer to validate.

    variable_name: str
       Name of the variable being validated, to increase
       the informativeness of the error message.

    Returns
    -------
    None
       If validation passes.

    Raises
    ------
    ValueError
       If validation fails.
    """
    if not isinstance(day_of_week, int):
        raise ValueError(
            "Day-of-week indices must be integers "
            "between 0 and 6, inclusive. "
            f"Got {day_of_week} for {variable_name}, "
            "which is a "
            f"{type(day_of_week)}"
        )
    if day_of_week < 0 or day_of_week > 6:
        raise ValueError(
            "Day-of-week indices must be a integers "
            "between 0 and 6, inclusive. "
            f"Got {day_of_week} for {variable_name}."
        )
    return None


def daily_to_weekly(
    daily_values: ArrayLike,
    input_data_first_dow: int = 0,
    week_start_dow: int = 0,
) -> ArrayLike:
    """
    Aggregate daily values (e.g.
    incident hospital admissions)
    to weekly total values.

    Parameters
    ----------
    daily_values : ArrayLike
        Daily timeseries values (e.g. incident infections or
        incident ed visits).
    input_data_first_dow : int
        First day of the week in the input timeseries `daily_values`.
        An integer between 0 and 6, inclusive (0 for Monday, 1 for Tuesday,
        ..., 6 for Sunday).
        If `input_data_first_dow` does not match `week_start_dow`, the
        incomplete first week is ignored and weekly values starting
        from the second week are returned. Defaults to 0.
    week_start_dow : int
        Day of the week on which weeks are considered to
        start in the output timeseries of weekly values
        (e.g. ISO weeks start on Mondays and end on Sundays;
        MMWR epiweeks start on Sundays and end on Saturdays).
        An integer between 0 and 6, inclusive (0 for Monday,
        1 for Tuesday, ..., 6 for Sunday).
        Default 0 (i.e. ISO weeks, starting on Mondays).

    Returns
    -------
    ArrayLike
        Data converted to weekly values starting
        with the first full week available.

    Raises
    ------
    ValueError
        If the specified days of the week fail validation.

    Notes
    -----
    This is _not_ a simple inverse of :func:`weekly_to_daily`.
    This function aggregates (by summing) daily values to
    create a timeseries of weekly total values.
    :func:`weekly_to_daily` broadcasts a _single shared value_
    for a given week as the (repeated) daily value for each day
    of that week.
    """

    validate_dow(input_data_first_dow, "input_data_first_dow")
    validate_dow(week_start_dow, "week_start_dow")

    offset = (week_start_dow - input_data_first_dow) % 7
    daily_values = daily_values[offset:]

    if len(daily_values) < 7:
        raise ValueError("No complete weekly values available")

    n_weeks = daily_values.shape[0] // 7
    trimmed = daily_values[:n_weeks * 7] 
    weekly_values = trimmed.reshape(n_weeks, 7, *daily_values.shape[1:]).sum(axis=1)

    return weekly_values


def daily_to_mmwr_epiweekly(
    daily_values: ArrayLike,
    input_data_first_dow: int = 6,
) -> ArrayLike:
    """
    Aggregate daily values to weekly values
    using :func:`daily_to_weekly` with
    MMWR epidemiological weeks (begin on Sundays,
    end on Saturdays).

    Parameters
    ----------
    daily_values : ArrayLike
        Daily timeseries values.
    input_data_first_dow : int
        First day of the week in the input timeseries `daily_values`.
        An integer between 0 and 6, inclusive (0 for Monday, 1 for
        Tuesday, ..., 6 for Sunday).
        If `input_data_first_dow` is _not_ the MMWR epiweek start day
        (6, Sunday), the incomplete first week is ignored and
        weekly values starting from the second week are returned.
        Defaults to 6 (Sunday).

    Returns
    -------
    ArrayLike
        Data converted to epiweekly values.
    """
    return daily_to_weekly(
        daily_values, input_data_first_dow, week_start_dow=6
    )


def weekly_to_daily(
    weekly_values: ArrayLike,
    week_start_dow: int = 0,
    output_data_first_dow: int = None,
) -> ArrayLike:
    """
    Broadcast a weekly timeseries to a daily
    timeseries. The value for the week will be used
    as the value each day in that week, via
    :func:`jnp.repeat`.

    Parameters
    ----------
    weekly_values: ArrayLike
        Timeseries of weekly values, where
        (discrete) time is the first dimension of
        the array (following Pyrenew convention).

    week_start_dow: int
        Day of the week on which weeks are considered to
        start in the input ``weekly_values`` timeseries
        (e.g. ISO weeks start on Mondays and end on Sundays;
        MMWR epiweeks start on Sundays and end on Saturdays).
        An integer between 0 and 6, inclusive (0 for Monday,
        1 for Tuesday, ..., 6 for Sunday).
        Default 0 (i.e. ISO weeks, starting on Mondays).

    output_data_first_dow: int
        Day of the week on which to start the output timeseries.
        An integer between 0 and 6, inclusive (0 for Monday,
        1 for Tuesday, ..., 6 for Sunday). Defaults to the week
        start date as specified by ``week_start_dow``.
        If ``output_data_first_dow`` is _not_ equal to ``week_start_dow``,
        the first weekly value will be partial (i.e. represented by
        between 1 and 6 entries in the output timeseries) and
        all subsequent weeks will be complete (represented by 7
        values each).

    Returns
    -------
    ArrayLike
        The daily timeseries.

    Raises
    ------
    ValueError
        If the specified days of the week fail validation.

    Notes
    -----
    This is _not_ a simple inverse of :func:`daily_to_weekly`.
    :func:`daily_to_weekly` aggregates (by summing) daily values to
    create a timeseries of weekly total values.
    This function broadcasts a _single shared value_
    for a given week as the (repeated) daily value for each day
    of that week.
    """

    validate_dow(week_start_dow, "week_start_dow")
    if output_data_first_dow is None:
        output_data_first_dow = week_start_dow
    validate_dow(output_data_first_dow, "output_data_first_dow")

    offset = (output_data_first_dow - week_start_dow) % 7
    return jnp.repeat(
        weekly_values,
        repeats=7,
        axis=0,
    )[offset:]


def mmwr_epiweekly_to_daily(
    weekly_values: ArrayLike,
    output_data_first_dow: int = 6,
) -> ArrayLike:
    """
    Convert an MMWR epiweekly timeseries to a daily
    timeseries using :func:`weekly_to_daily`.

    Parameters
    ----------
    weekly_values: ArrayLike
        Timeseries of weekly values, where
        (discrete) time is the first dimension of
        the array (following Pyrenew convention).

    output_data_first_dow: int
        Day of the week on which to start the output timeseries.
        An integer between 0 and 6, inclusive (0 for Monday,
        1 for Tuesday, ..., 6 for Sunday). Defaults to the MMWR
        epiweek start day (6, Sunday).
        If ``output_data_first_dow`` is _not_ equal to 6 (Sunday,
        the start of an MMWR epiweek), the first weekly value will
        be partial (i.e. represented by between 1 and 6 entries
        in the output timeseries) and all subsequent weeks will be
        complete (represented by 7 values each).

    Returns
    -------
    ArrayLike
        The daily timeseries.
    """
    return weekly_to_daily(
        weekly_values,
        output_data_first_dow=output_data_first_dow,
        week_start_dow=6,
    )
