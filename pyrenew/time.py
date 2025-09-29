"""
Helper functions for handling timeseries in Pyrenew

Days of the week in pyrenew are 0-indexed and follow
ISO standards, so 0 is Monday at 6 is Sunday.
"""
import datetime as dt
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import polars as pl
from typing import Union, Tuple


def validate_dow(day_of_week: int, variable_name: str) -> None:
    """
    Confirm that an integer is a valid Pyrenew day of the week
    index, with informative error messages on failure.

    Parameters
    ----------
    day_of_week
        Integer to validate.

    variable_name
        Name of the variable being validated, to increase the informativeness of the error message.

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


def validate_mmwr_dates(dates: ArrayLike) -> None:
    """
    Validate that dates are Saturdays (MMWR week endings).

    :param dates: Array of dates to validate
    :raises ValueError: If any date is not a Saturday
    """
    for date in dates:
        if date is None:  # Skip None values
            continue
        if isinstance(date, np.datetime64):
            date = date.astype(dt.datetime)
        if date.weekday() != 5:  # Saturday
            raise ValueError(
                f"MMWR dates must be Saturdays (weekday=5). "
                f"Got {date.strftime('%A')} ({date.weekday()}) for {date}"
            )


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
    daily_values
        Daily timeseries values (e.g. incident infections or
        incident ed visits).
    input_data_first_dow
        First day of the week in the input timeseries `daily_values`.
        An integer between 0 and 6, inclusive (0 for Monday, 1 for Tuesday,
        ..., 6 for Sunday).
        If `input_data_first_dow` does not match `week_start_dow`, the
        incomplete first week is ignored and weekly values starting
        from the second week are returned. Defaults to 0.
    week_start_dow
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
    This is _not_ a simple inverse of [`pyrenew.time.weekly_to_daily`][].
    This function aggregates (by summing) daily values to
    create a timeseries of weekly total values.
    [`pyrenew.time.weekly_to_daily`][] broadcasts a _single shared value_
    for a given week as the (repeated) daily value for each day
    of that week.
    """

    validate_dow(input_data_first_dow, "input_data_first_dow")
    validate_dow(week_start_dow, "week_start_dow")

    offset = (week_start_dow - input_data_first_dow) % 7
    daily_values = daily_values[offset:]

    if daily_values.shape[0] < 7:
        raise ValueError("No complete weekly values available")

    n_weeks = daily_values.shape[0] // 7
    trimmed = daily_values[: n_weeks * 7]
    weekly_values = trimmed.reshape(n_weeks, 7, *daily_values.shape[1:]).sum(axis=1)

    return weekly_values


def daily_to_mmwr_epiweekly(
    daily_values: ArrayLike,
    input_data_first_dow: int = 6,
) -> ArrayLike:
    """
    Aggregate daily values to weekly values
    using [`pyrenew.time.daily_to_weekly`][] with
    MMWR epidemiological weeks (begin on Sundays,
    end on Saturdays).

    Parameters
    ----------
    daily_values
        Daily timeseries values.
    input_data_first_dow
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
    return daily_to_weekly(daily_values, input_data_first_dow, week_start_dow=6)


def weekly_to_daily(
    weekly_values: ArrayLike,
    week_start_dow: int = 0,
    output_data_first_dow: int = None,
) -> ArrayLike:
    """
    Broadcast a weekly timeseries to a daily
    timeseries. The value for the week will be used
    as the value each day in that week, via
    [`jax.numpy.repeat`][].

    Parameters
    ----------
    weekly_values
        Timeseries of weekly values, where
        (discrete) time is the first dimension of
        the array (following Pyrenew convention).

    week_start_dow
        Day of the week on which weeks are considered to
        start in the input `weekly_values` timeseries
        (e.g. ISO weeks start on Mondays and end on Sundays;
        MMWR epiweeks start on Sundays and end on Saturdays).
        An integer between 0 and 6, inclusive (0 for Monday,
        1 for Tuesday, ..., 6 for Sunday).
        Default 0 (i.e. ISO weeks, starting on Mondays).

    output_data_first_dow
        Day of the week on which to start the output timeseries.
        An integer between 0 and 6, inclusive (0 for Monday,
        1 for Tuesday, ..., 6 for Sunday). Defaults to the week
        start date as specified by `week_start_dow`.
        If `output_data_first_dow` is _not_ equal to `week_start_dow`,
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
    This is _not_ a simple inverse of [`pyrenew.time.daily_to_weekly`][].
    [`pyrenew.time.daily_to_weekly`][] aggregates (by summing) daily values to
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
    timeseries using [`pyrenew.time.weekly_to_daily`][].

    Parameters
    ----------
    weekly_values
        Timeseries of weekly values, where
        (discrete) time is the first dimension of
        the array (following Pyrenew convention).

    output_data_first_dow
        Day of the week on which to start the output timeseries.
        An integer between 0 and 6, inclusive (0 for Monday,
        1 for Tuesday, ..., 6 for Sunday). Defaults to the MMWR
        epiweek start day (6, Sunday).
        If `output_data_first_dow` is _not_ equal to 6 (Sunday,
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


def date_to_model_t(
    date: Union[dt.datetime, np.datetime64],
    start_date: Union[dt.datetime, np.datetime64]
) -> int:
    """
    Convert calendar date to model time index.

    :param date: Target date
    :param start_date: Date corresponding to model time t=0
    :return: Model time index (days since start_date)
    """
    if isinstance(date, np.datetime64):
        date = date.astype('datetime64[D]').astype(object)
    elif isinstance(date, dt.datetime):
        date = date.date()
    
    if isinstance(start_date, np.datetime64):
        start_date = start_date.astype('datetime64[D]').astype(object)
    elif isinstance(start_date, dt.datetime):
        start_date = start_date.date()

    return (date - start_date).days

def model_t_to_date(
    model_t: int,
    start_date: Union[dt.datetime, np.datetime64]
) -> dt.datetime:
    """
    Convert model time index to calendar date.

    :param model_t: Model time index
    :param start_date: Date corresponding to model time t=0
    :return: Calendar date
    """
    if isinstance(start_date, np.datetime64):
        start_date = start_date.astype('datetime64[D]').astype(object)
    
    # Ensure we have datetime, not just date
    if isinstance(start_date, dt.date) and not isinstance(start_date, dt.datetime):
        start_date = dt.datetime.combine(start_date, dt.time())

    return start_date + dt.timedelta(days=model_t)


def get_observation_indices(
    observed_dates: ArrayLike,
    data_start_date: Union[dt.datetime, np.datetime64],
    freq: str = "mmwr_weekly"
) -> jnp.ndarray:
    """
    Get indices for observed data in aggregated time series.

    :param observed_dates: Dates of observations
    :param data_start_date: Start date of the data series
    :param freq: Frequency of aggregated data ("mmwr_weekly" or "weekly")
    :return: Indices for observed data points in aggregated series
    :raises NotImplementedError: For unsupported frequencies
    """
    if isinstance(data_start_date, np.datetime64):
        data_start_date = data_start_date.astype(dt.datetime)

    if freq == "mmwr_weekly":
        # Calculate weeks since first Saturday (MMWR week end)
        days_to_first_saturday = (5 - data_start_date.weekday()) % 7
        if days_to_first_saturday == 0 and data_start_date.weekday() != 5:
            days_to_first_saturday = 7
        first_saturday = data_start_date + dt.timedelta(days=days_to_first_saturday)

        indices = []
        for obs_date in observed_dates:
            if isinstance(obs_date, np.datetime64):
                obs_date = obs_date.astype(dt.datetime)
            weeks_diff = (obs_date - first_saturday).days // 7
            indices.append(weeks_diff)
        return jnp.array(indices)

    elif freq == "weekly":
        # Calculate weeks since first Monday (ISO week start)
        days_to_first_monday = (7 - data_start_date.weekday()) % 7
        if days_to_first_monday == 0 and data_start_date.weekday() != 0:
            days_to_first_monday = 7
        first_monday = data_start_date + dt.timedelta(days=days_to_first_monday)

        indices = []
        for obs_date in observed_dates:
            if isinstance(obs_date, np.datetime64):
                obs_date = obs_date.astype(dt.datetime)
            weeks_diff = (obs_date - first_monday).days // 7
            indices.append(weeks_diff)
        return jnp.array(indices)

    else:
        raise NotImplementedError(f"Frequency '{freq}' not implemented")


def get_date_range_length(date_array: ArrayLike, timestep_days: int = 1) -> int:
    """Calculate number of time steps in a date range."""
    return (
        (max(date_array) - min(date_array))
        // np.timedelta64(timestep_days, "D")
        + 1
    ).item()


def aggregate_with_dates(
    daily_data: ArrayLike,
    start_date: Union[dt.datetime, np.datetime64],
    target_freq: str = "mmwr_weekly"
) -> Tuple[jnp.ndarray, dt.datetime]:
    """
    Aggregate daily data with automatic date handling.

    :param daily_data: Daily time series
    :param start_date: Date of first data point
    :param target_freq: Target frequency ("mmwr_weekly" or "weekly")
    :return: (aggregated_data, first_aggregated_date)
    :raises ValueError: For unsupported frequencies
    """
    if isinstance(start_date, np.datetime64):
        start_date = start_date.astype(dt.datetime)

    if target_freq == "mmwr_weekly":
        # Convert to PyRenew day-of-week indexing (0=Monday, 6=Sunday)
        first_dow = (start_date.weekday() + 1) % 7
        if first_dow == 0:  # Monday in datetime becomes 6 in PyRenew
            first_dow = 6
        else:
            first_dow = first_dow - 1

        weekly_data = daily_to_mmwr_epiweekly(daily_data, first_dow)

        # Calculate first Saturday (MMWR week end)
        days_to_saturday = (5 - start_date.weekday()) % 7
        if days_to_saturday == 0 and start_date.weekday() != 5:
            days_to_saturday = 7
        first_weekly_date = start_date + dt.timedelta(days=days_to_saturday)

    elif target_freq == "weekly":
        # Convert to PyRenew indexing
        first_dow = (start_date.weekday() + 1) % 7
        if first_dow == 0:
            first_dow = 6
        else:
            first_dow = first_dow - 1

        weekly_data = daily_to_weekly(daily_data, first_dow, week_start_dow=0)

        # Calculate first Monday (ISO week start)
        days_to_monday = (7 - start_date.weekday()) % 7
        if days_to_monday == 0 and start_date.weekday() != 0:
            days_to_monday = 7
        first_weekly_date = start_date + dt.timedelta(days=days_to_monday)

    else:
        raise ValueError(f"Unsupported target frequency: {target_freq}")

    return weekly_data, first_weekly_date


def create_date_time_spine(
    start_date: Union[dt.datetime, np.datetime64],
    end_date: Union[dt.datetime, np.datetime64],
    freq: str = "1d"
) -> pl.DataFrame:
    """
    Create a DataFrame mapping calendar dates to model time indices.

    :param start_date: First date (becomes t=0)
    :param end_date: Last date
    :param freq: Frequency string for polars date_range
    :return: DataFrame with 'date' and 't' columns
    """
    # Convert np.datetime64 to datetime.date for polars compatibility
    if isinstance(start_date, np.datetime64):
        start_date = start_date.astype('datetime64[D]').astype(dt.datetime)
    elif isinstance(start_date, dt.datetime):
        start_date = start_date.date()

    if isinstance(end_date, np.datetime64):
        end_date = end_date.astype('datetime64[D]').astype(dt.datetime)
    elif isinstance(end_date, dt.datetime):
        end_date = end_date.date()

    return (
        pl.DataFrame({
            "date": pl.date_range(
                start=start_date,
                end=end_date,
                interval=freq,
                eager=True,
            )
        })
        .with_row_index("t")
        .with_columns(pl.col("t").cast(pl.Int64))
    )

def get_end_date(
    start_date: Union[dt.datetime, np.datetime64],
    n_points: int,
    timestep_days: int = 1
) -> Union[np.datetime64, None]:
    """
    Calculate end date from start date and number of data points.

    :param start_date: First date in the series
    :param n_points: Number of data points
    :param timestep_days: Days between consecutive points
    :return: Date of the last data point
    :raises ValueError: If n_points is non-positive
    """
    if start_date is None:
        if n_points > 0:
            raise ValueError(
                f"Must provide start_date if n_points > 0. "
                f"Got n_points={n_points} with start_date=None"
                )
        return None

    if n_points < 0:
        raise ValueError(f"n_points must be non-negative, got {n_points}")

    if isinstance(start_date, dt.datetime):
        start_date = np.datetime64(start_date.date())

    return start_date + np.timedelta64((n_points - 1) * timestep_days, "D")


def get_n_data_days(
    n_points: int = None,
    date_array: ArrayLike = None,
    timestep_days: int = 1
) -> int:
    """
    Determine data length from either point count or date array.

    :param n_points: Explicit number of data points
    :param date_array: Array of observation dates
    :param timestep_days: Days between consecutive points
    :return: Number of data points
    :raises ValueError: If both or neither arguments provided
    """
    if n_points is None and date_array is None:
        return 0
    elif date_array is not None and n_points is not None:
        raise ValueError(
            "Must provide at most one of n_points and date_array"
        )
    elif date_array is not None:
        return get_date_range_length(date_array, timestep_days)
    else:
        return n_points


def align_observation_times(
    observation_dates: ArrayLike,
    model_start_date: Union[dt.datetime, np.datetime64],
    aggregation_freq: str = "daily"
) -> jnp.ndarray:
    """
    Convert observation dates to model time indices with temporal aggregation.

    :param observation_dates: Dates when observations occurred
    :param model_start_date: Date corresponding to model time t=0
    :param aggregation_freq: Temporal aggregation ("daily", "weekly", "mmwr_weekly")
    :return: Model time indices for observations
    :raises NotImplementedError: For unsupported frequencies
    """
    if aggregation_freq == "daily":
        return jnp.array([
            date_to_model_t(date, model_start_date)
            for date in observation_dates
        ])
    elif aggregation_freq in ["weekly", "mmwr_weekly"]:
        return get_observation_indices(
            observation_dates, model_start_date, aggregation_freq
        )
    else:
        raise NotImplementedError(f"Frequency '{aggregation_freq}' not supported")
