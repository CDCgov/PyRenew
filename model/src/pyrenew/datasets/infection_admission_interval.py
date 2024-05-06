from importlib.resources import files

import polars as pl


def load_infection_admission_interval() -> pl.DataFrame:
    """
    Load the infection to admission interval

    This dataset contains the infection to admission interval distribution for
    COVID-19.

    Returns
    -------
    pl.DataFrame
        The infection to admission interval dataset

    Notes
    -----
    This dataset was downloaded directly from:
    https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/inf_to_hosp.csv

    The dataset contains the following columns:
        - `timepoint`
        - `probability_mass`
    """

    # Load the dataset
    return pl.read_csv(
        source=files("pyrenew.datasets") / "infection_admission_interval.tsv",
        separator="\t",
    )
