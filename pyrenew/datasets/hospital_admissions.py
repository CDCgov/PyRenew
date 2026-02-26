# numpydoc ignore=ES01,SA01,EX01
"""
Load hospital admissions data for use in tutorials and examples.

This module provides functions to load COVID-19 hospital admissions
data (daily and weekly) from the CDC's cfa-forecast-renewal-ww project.
"""

from importlib.resources import files

import jax.numpy as jnp
import polars as pl


def load_hospital_data_for_state(
    state_abbr: str = "CA",
    filename: str = "2023-11-06.csv",
) -> dict:
    """
    Load hospital admissions data for a specific state.

    Parameters
    ----------
    state_abbr : str
        State abbreviation (e.g., "CA"). Default is "CA".
    filename : str
        CSV filename. Default is "2023-11-06.csv".

    Returns
    -------
    dict
        Dictionary containing:

        - daily_admits: JAX array of daily hospital admissions
        - population: Population size (scalar)
        - dates: List of datetime.date objects
        - n_days: Number of days

    Notes
    -----
    Data source: CDC cfa-forecast-renewal-ww repository.
    License: Public Domain (CC0 1.0 Universal) - U.S. Government work.
    """
    data_path = files("pyrenew.datasets.hospital_admissions_data") / filename
    df = pl.read_csv(source=data_path)

    df = (
        df.with_columns(pl.col("date").str.to_date())
        .filter(pl.col("location") == state_abbr)
        .sort("date")
    )

    if len(df) == 0:
        raise ValueError(f"No data found for state {state_abbr} in {filename}")

    daily_admits = jnp.array(df["daily_hosp_admits"].to_numpy())
    population = int(df["pop"][0])
    dates = df["date"].to_list()

    return {
        "daily_admits": daily_admits,
        "population": population,
        "dates": dates,
        "n_days": len(daily_admits),
    }


def load_weekly_hospital_data_for_state(
    state_abbr: str = "CA",
    filename: str = "2023-11-06_weekly.csv",
) -> dict:
    """
    Load weekly (epiweek) hospital admissions data for a specific state.

    Parameters
    ----------
    state_abbr : str
        State abbreviation (e.g., "CA"). Default is "CA".
    filename : str
        CSV filename. Default is "2023-11-06_weekly.csv".

    Returns
    -------
    dict
        Dictionary containing:

        - weekly_admits: JAX array of weekly hospital admissions
        - population: Population size (scalar)
        - week_ends: List of datetime.date objects (epiweek ending dates)
        - n_weeks: Number of weeks

    Notes
    -----
    Data source: aggregated from daily CDC cfa-forecast-renewal-ww data.
    License: Public Domain (CC0 1.0 Universal) - U.S. Government work.
    """
    data_path = files("pyrenew.datasets.hospital_admissions_data") / filename
    df = pl.read_csv(source=data_path)

    df = (
        df.with_columns(pl.col("week_end").str.to_date())
        .filter(pl.col("location") == state_abbr)
        .sort("week_end")
    )

    if len(df) == 0:
        raise ValueError(f"No data found for state {state_abbr} in {filename}")

    weekly_admits = jnp.array(df["weekly_hosp_admits"].to_numpy())
    population = int(df["pop"][0])
    week_ends = df["week_end"].to_list()

    return {
        "weekly_admits": weekly_admits,
        "population": population,
        "week_ends": week_ends,
        "n_weeks": len(weekly_admits),
    }
