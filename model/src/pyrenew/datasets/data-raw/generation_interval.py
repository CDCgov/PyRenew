# Dataset from:
# https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/generation_interval.csv
import os

import polars as pl

gen_int = pl.read_csv(
    "https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/generation_interval.csv",
)

# Building path to save the file
path = os.path.join(
    "src",
    "pyrenew",
    "datasets",
    "gen_int.tsv",
)

assert os.path.exists(os.path.dirname(path))

gen_int.write_csv(
    file=path,
    separator="\t",
    include_header=True,
    null_value="",
)
