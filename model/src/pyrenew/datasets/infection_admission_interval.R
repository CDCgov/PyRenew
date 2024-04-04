# Dataset from:
# https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/inf_to_hosp.csv
infection_admission_interval <- read.csv("https://raw.githubusercontent.com/CDCgov/wastewater-informed-covid-forecasting/0962c5d1652787479ac72caebf076ab55fe4e10c/input/saved_pmfs/inf_to_hosp.csv")

# Saving as TSV
write.table(
  x = infection_admission_interval,
  file = "src/pyrenew/datasets/infection_admission_interval.tsv",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
  )
