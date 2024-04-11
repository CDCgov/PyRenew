# Dataset from:
# https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/generation_interval.csv
import polars as pl

gen_int = pl.read_csv(
    "https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/generation_interval.csv",
)

gen_int.write_csv(
    file="src/pyrenew/datasets/gen_int.tsv",
    separator="\t",
    include_header=True,
    null_value="",
)
