# numpydoc ignore=ES01,SA01,EX01
"""
Load wastewater NWSS data for use in tutorials and examples.

This module provides functions to load synthetic wastewater surveillance
data in NWSS format from the CDC's cfa-forecast-renewal-ww project.
"""

from importlib.resources import files

import jax.numpy as jnp
import polars as pl


def load_wastewater_data_for_state(
    state_abbr: str = "CA",
    filename: str = "fake_nwss.csv",
) -> dict:
    """
    Load wastewater data for a specific state.

    Parameters
    ----------
    state_abbr
        State abbreviation (e.g., "CA"). Default is "CA".
    filename
        CSV filename. Default is "fake_nwss.csv".

    Returns
    -------
    dict
        Dictionary containing:

        - observed_conc: JAX array of log concentrations (log copies/mL)
        - observed_conc_linear: JAX array of linear concentrations (copies/mL)
        - site_ids: JAX array of site indices
        - time_indices: JAX array of time indices (days from start)
        - wwtp_names: List of unique WWTP names
        - dates: List of unique dates
        - n_sites: Number of unique sites
        - n_obs: Number of observations
        - raw_df: Polars DataFrame (for debugging)

    Notes
    -----
    Data source: CDC cfa-forecast-renewal-ww repository.
    License: Public Domain (CC0 1.0 Universal) - U.S. Government work.

    The data is synthetic and contains deliberately added noise for
    public release. Concentrations are in copies/L and are converted
    to copies/mL (divided by 1000).
    """
    data_path = files("pyrenew.datasets.wastewater_nwss_data") / filename
    df = pl.read_csv(
        source=data_path,
        schema_overrides={"county_names": pl.String},
    )
    df = df.with_columns(pl.col("sample_collect_date").str.to_date())

    # Filter to requested state
    df = df.filter(pl.col("wwtp_jurisdiction") == state_abbr)
    if len(df) == 0:
        raise ValueError(f"No wastewater data found for state {state_abbr}")

    # Convert copies/L to copies/mL
    df = df.with_columns(
        (pl.col("pcr_target_avg_conc") / 1000).alias("conc_linear"),
    )

    df = df.sort("sample_collect_date")
    unique_sites = sorted(df["wwtp_name"].unique().to_list())
    site_to_idx = {site: idx for idx, site in enumerate(unique_sites)}
    min_date = df["sample_collect_date"].min()
    df = df.with_columns(
        ((pl.col("sample_collect_date") - min_date).dt.total_days()).alias("time_idx")
    )
    df = df.with_columns(
        pl.col("wwtp_name").replace_strict(site_to_idx, default=None).alias("site_idx")
    )

    observed_conc_linear = jnp.array(df["conc_linear"].to_numpy())
    observed_log_conc = jnp.log(observed_conc_linear + 1e-8)
    site_ids = jnp.array(df["site_idx"].to_numpy(), dtype=jnp.int32)
    time_indices = jnp.array(df["time_idx"].to_numpy(), dtype=jnp.int32)

    return {
        "observed_conc": observed_log_conc,
        "observed_conc_linear": observed_conc_linear,
        "site_ids": site_ids,
        "time_indices": time_indices,
        "wwtp_names": unique_sites,
        "dates": sorted(df["sample_collect_date"].unique().to_list()),
        "n_sites": len(unique_sites),
        "n_obs": len(df),
        "raw_df": df,
    }
