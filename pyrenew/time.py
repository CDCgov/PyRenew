"""
Helper functions for handling timeseries in Pyrenew
"""


def daily_to_weekly(
    daily_values: ArrayLike,
    input_data_first_dow: int = 0,
    week_start_dow: int = 0,
) -> ArrayLike:
    """
    Aggregate daily values (e.g.
    incident hospital admissions) into weekly total values.

    Parameters
    ----------
    daily_values : ArrayLike
        Daily timeseries values (e.g. incident infections or incident ed visits).
    input_data_first_dow : int
        First day of the week in the input timeseries `daily_values`.
        An integer between 0 and 6, inclusive (0 for Monday, 6 for Sunday).
        If `input_data_first_dow` does not match `week_start_dow`, the incomplete first
        week is ignored and weekly values starting
        from the second week are returned. Defaults to 0.
    week_start_dow : int
        The desired starting day of the week for the output weekly aggregation.
        An integer between 0 and 6, inclusive. Defaults to 0 (Monday).

    Returns
    -------
    ArrayLike
        Data converted to weekly values starting
        with the first full week available.
    """
    if input_data_first_dow < 0 or input_data_first_dow > 6:
        raise ValueError(
            "First day of the week for input timeseries must be between 0 and 6."
        )

    if week_start_dow < 0 or week_start_dow > 6:
        raise ValueError(
            "Week start date for output aggregated values must be between 0 and 6."
        )

    offset = (week_start_dow - input_data_first_dow) % 7
    daily_values = daily_values[offset:]

    if len(daily_values) < 7:
        raise ValueError("No complete weekly values available")

    weekly_values = jnp.convolve(daily_values, jnp.ones(7), mode="valid")[::7]

    return weekly_values


def daily_to_mmwr_epiweekly(
    daily_values: ArrayLike, input_data_first_dow: int = 0
) -> ArrayLike:
    """
    Convert daily values to MMWR epidemiological weeks.

    Parameters
    ----------
    daily_values : ArrayLike
        Daily timeseries values.
    input_data_first_dow : int
        First day of the week in the input timeseries `daily_values`.
        Defaults to 0 (Monday).

    Returns
    -------
    ArrayLike
        Data converted to epiweekly values.
    """
    return daily_to_weekly(
        daily_values, input_data_first_dow, week_start_dow=6
    )


def weekly_to_daily(
    weekly_ts: ArrayLike, first_day_dow: int = 0, week_start_dow: int = 0
) -> ArrayLike:
    """
    Convert a weekly timeseries to a daily
    timeseries using :func:`jnp.repeat`.

    Parameters
    ----------
    weekly_ts: ArrayLike
        Timeseries of weekly values, where
        (discrete) time is the first dimension of
        the array (following Pyrenew convention).

    first_day_dow: int
        First day of the week in the daily timeseries.
        0-indexed. Default 0.

    week_start_dow: int
        Starting day of the week for ``weekly_ts``,
        0-indexed. Default 0.

    Returns
    -------
    ArrayLike
        The daily timeseries.
    """
    first_ind = (first_day_dow - week_start_dow) % 7
    return jnp.repeat(
        weekly_ts,
        repeats=7,
        axis=0,
    )[first_ind:]
