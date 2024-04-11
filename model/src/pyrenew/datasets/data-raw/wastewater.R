load(
  "https://github.com/CDCgov/wastewater-informed-covid-forecasting/blob/292526383ece582f10823fc939c7e590ca349c6d/cfaforecastrenewalww/data/example_df.rda"
  )

# Saving as TSV
write.table(
  x = example_df,
  file = "src/pyrenew/datasets/wastewater.tsv",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
  )
