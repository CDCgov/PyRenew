load("src/pyrenew/datasets/example_df.rda")

# Saving as TSV
write.table(
  x = example_df,
  file = "src/pyrenew/datasets/wastewater.tsv",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
  )
