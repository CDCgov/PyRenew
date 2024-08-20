# Dataset from:
# https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/inf_to_hosp.csv

# numpydoc ignore=GL08

import os

import polars as pl

# Read CSV file from URL
url = "https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/inf_to_hosp.csv"
infection_admission_interval = pl.read_csv(url)

# Building path to save the file
path = os.path.join(
    "src",
    "pyrenew",
    "datasets",
    "infection_admission_interval.tsv",
)

os.makedirs(os.path.dirname(path), exist_ok=True)

# Save as TSV
infection_admission_interval.write_csv(
    file=path,
    separator="\t",
    include_header=True,
    null_value="",
)
