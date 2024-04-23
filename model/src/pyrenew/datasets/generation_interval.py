from importlib.resources import files

import polars as pl


def load_generation_interval() -> pl.DataFrame:
    """Load the generation interval dataset

    This dataset contains the generation interval distribution for COVID-19.

    Returns
    -------
    pl.DataFrame
        The generation interval dataset

    Notes
    -----

    This dataset was downloaded directly from:
    https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/generation_interval.csv

    The dataset contains the following columns:
        - `timepoint`
        - `probability_mass`
    """

    # Load the dataset
    return pl.read_csv(
        source=files("pyrenew.datasets") / "generation_interval.tsv",
        separator="\t",
    )
