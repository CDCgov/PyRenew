# Run this after running the `toy_data_vignette.Rmd` vignette
# https://github.com/CDCgov/wastewater-informed-covid-forecasting/blob/prod/cfaforecastrenewalww/vignettes/toy_data_vignette.Rmd
library(jsonlite)
library(fs)

base_dir <- path("scratch")


write_json(stan_data, path(base_dir, "stan_data", ext = "json"))
write_json(stan_data_hosp_only, path(base_dir, "stan_data_hosp_only", ext = "json"))

vignette_ww_model_dir <- path(base_dir, "vignette_ww_model")
cfaforecastrenewalww::create_dir(vignette_ww_model_dir)

fit_dynamic_rt$save_output_files(
  dir = vignette_ww_model_dir
)


vignette_hosp_only_model_dir <- path(base_dir, "vignette_hosp_only_model")
cfaforecastrenewalww::create_dir(vignette_hosp_only_model_dir)

fit_dynamic_rt_hosp_only$save_output_files(
  dir = vignette_hosp_only_model_dir
)
