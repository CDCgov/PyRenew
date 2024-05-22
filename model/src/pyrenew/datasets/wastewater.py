# numpydoc ignore=ES01,SA01,EX01

"""
This module loads the package dataset named 'wastewater' and provides functions to manipulate the data. It uses the 'polars' library.
"""


from importlib.resources import files

import polars as pl


def load_wastewater() -> pl.DataFrame:  # numpydoc ignore=SS06,SA01,EX01
    """
    Load the wastewater dataset. This dataset
    contains simulated entries of
    COVID-19 wastewater concentration data.
    The dataset is used to demonstrate the use of
    the wastewater-informed COVID-19 forecasting model.

    Returns
    -------
    pl.DataFrame
        The wastewater dataset.

    Notes
    -----
    This dataset was downloaded directly from:
    https://github.com/CDCgov/wastewater-informed-covid-forecasting/blob/292526383ece582f10823fc939c7e590ca349c6d/cfaforecastrenewalww/data/example_df.rda

    The dataset contains the following columns:
        - `lab_wwtp_unique_id`
        - `log_conc`
        - `date`
        - `load_sewage`
        - `below_load`
        - `daily_hosp_admits`
        - `daily_hosp_admits_for_eval`
        - `pop`
        - `forecast_date`
        - `hosp_calibration_time`
        - `site`
        - `ww_pop`
        - `inf_per_capita`
    """

    # Load the dataset
    return pl.read_csv(
        source=files("pyrenew.datasets") / "wastewater.tsv",
        separator="\t",
    )
