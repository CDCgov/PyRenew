# Dataset from:
# https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/generation_interval.csv
gen_int <- read.csv("https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/generation_interval.csv")

# Saving as TSV
write.table(
  x = gen_int,
  file = "src/pyrenew/datasets/gen_int.tsv",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
  )
