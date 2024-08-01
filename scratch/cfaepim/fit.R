#!/usr/bin/env Rscript

#' fit forecast models for a given report date
#'
#' @param report_date report date for which
#' to run the analysis
#' @param output_parent_directory report will
#' be saved in a subdirectory named after the report date,
#' but within this parent directory. Defaults to creating
#' and/or using a directory named `"output"` within the
#' current working directory for this purpose.
#' @param data_cutoff_date Unless use data through
#' the given date. If `NULL`, use all
#' available data. Default `NULL`.
#' @param locations Only fit these locations.
#' If `NULL`, use all available locations.
#' Default `NULL`.
#' @param param_path Where to look for a parameter
#' file. Default to a file named `"params.toml"`
#' within a directory named `"data"` within the
#' current working directory.
#' @param location_data_path Where to look for a FluSight
#' `locations.csv` containing locations to fit and their
#' populations. Default to a file named `"locations.csv"`
#' within a directory named `"data"` within the
#' current working directory.
#' @param healthdata_api_key_id API key ID for authenticating
#' to HealthData.gov SODA API. Not required, but polite.
#' Default `NULL`
#' @param healthdata_api_key_secret Corresponding
#' API key secrete for authenticating
#' to HealthData.gov SODA API. Not required, but polite.
#' Default `NULL`.
#' @param overwrite_params Overwrite an existing
#' archived parameter file if it exists?
#' Boolean, default `FALSE`. If `FALSE`
#' and an archived parameter file already
#' exists, the pipeline will error out.
#' @return `TRUE` on success.
fit <- function(report_date,
                output_parent_directory = "output",
                data_cutoff_date = NULL,
                locations = NULL,
                param_path = fs::path("data", "params.toml"),
                location_data_path = fs::path("data", "locations.csv"),
                healthdata_api_key_id = NULL,
                healthdata_api_key_secret = NULL,
                overwrite_params = FALSE) {
  cli::cli_inform("Using working directory {fs::path_wd()}")

  report_outdir <- fs::path(
    output_parent_directory,
    report_date
  )

  fs::dir_create(report_outdir)

  data_save_path <- fs::path(
    report_outdir,
    paste0(report_date, "_clean_data", ".tsv")
  )

  param_save_path <- fs::path(
    report_outdir,
    paste0(report_date, "_config", ".toml")
  )

  cli::cli_inform("reading in run parameters from {param_path}")
  params <- RcppTOML::parseTOML(param_path)

  cli::cli_inform("Archiving parameters at {param_save_path}")
  fs::file_copy(param_path,
    param_save_path,
    overwrite = overwrite_params
  )

  cli::cli_inform("Pulling and cleaning data")
  clean_data <- cfaepim::get_data(
    params$first_fitting_date,
    location_data_path,
    api_key_id = healthdata_api_key_id,
    api_key_secret = healthdata_api_key_secret,
    recency_effect_length = params$recency_effect_length
  )

  for (loc in unique(clean_data$location)) {
    loc_start_date <- params$location_specific_start_dates[[loc]]
    loc_cutoff_date <- params$location_specific_cutoff_dates[[loc]]

    if (!is.null(loc_start_date)) {
      cli::cli_inform(paste0(
        "Using custom start date {loc_start_date} ",
        "for location {loc}"
      ))
      clean_data <- clean_data |>
        dplyr::filter(location != !!loc | date >= !!loc_start_date)
    }

    if (!is.null(loc_cutoff_date)) {
      cli::cli_inform(paste0(
        "Using custom cutoff date {loc_cutoff_date} ",
        "for location {loc}"
      ))
      clean_data <- clean_data |>
        dplyr::filter(location != !!loc | date <= !!loc_cutoff_date)
    }
  }

  if (!is.null(data_cutoff_date)) {
    clean_data <- clean_data |>
      dplyr::filter(date <= data_cutoff_date)
  }

  unobserved_dates <- params$location_specific_excluded_dates |>
    stack() |>
    tibble::as_tibble() |>
    dplyr::mutate(
      date = as.Date(values),
      location = ind,
      nonobservation_period = TRUE
    ) |>
    dplyr::select(
      date,
      location,
      nonobservation_period
    )

  clean_data <- clean_data |>
    dplyr::left_join(
      unobserved_dates,
      by = c("location", "date")
    ) |>
    dplyr::mutate(
      nonobservation_period =
        tidyr::replace_na(
          nonobservation_period,
          FALSE
        )
    )


  cli::cli_inform("Archiving cleaned data at {data_save_path}")
  readr::write_tsv(clean_data, data_save_path)

  if (!is.null(locations)) {
    loc_vec <- as.character(locations)
  } else {
    loc_vec <- clean_data |>
      dplyr::distinct(location) |>
      dplyr::pull()
  }
  names(loc_vec) <- loc_vec

  cli::cli_alert("Fitting the following locations: {loc_vec}")

  cli::cli_alert("Setting up models")
  fitting_args <- lapply(loc_vec,
    cfaepim::build_state_light_model,
    clean_data = clean_data,
    params = params,
    adapt_delta = params$mcmc$adapt_delta,
    max_treedepth = params$mcmc$max_treedepth,
    n_chains = params$mcmc$n_chains,
    n_warmup = params$mcmc$n_warmup,
    n_iter = params$mcmc$n_iter
  )

  cli::cli_alert("{length(fitting_args)} models to fit")
  cli::cli_alert("Starting model fit at {Sys.time()}")

  raw_results <- cfaepim::fit_future(
    fitting_args,
    save_results = TRUE,
    overwrite_existing = FALSE,
    save_dir = report_outdir,
    save_filename_pattern = paste0("_", report_date, "_epim_results")
  )

  print(raw_results[[1]])

  cli::cli_alert("Model fit finished at {Sys.time()}")

  return(TRUE)
}

argv_parser <- argparser::arg_parser(
  paste0(
    "Run Epidemia forecast analysis ",
    "for a given report date"
  )
) |>
  argparser::add_argument(
    "report_date",
    help = "Date for which to generate a forecast report"
  ) |>
  argparser::add_argument(
    "--data-cutoff",
    help = "Only use data up to this date for forecasting"
  ) |>
  argparser::add_argument(
    "--locations",
    help = "Only fit to these locations"
  ) |>
  argparser::add_argument(
    "--outdir",
    help = paste0(
      "Write forecast output to a timestamped ",
      "subdirectory of this directory"
    ),
    default = "output"
  ) |>
  argparser::add_argument(
    "--params",
    help = "Path to parameter file",
    default = "data/params.toml"
  ) |>
  argparser::add_argument(
    "--overwrite-params",
    help = "Overwrite an existing archived parameter file?",
    default = FALSE
  )

argv <- argparser::parse_args(argv_parser)

n_cores_use <- parallel::detectCores() - 1
future::plan(future::multicore(workers = n_cores_use))

if (is.na(argv$data_cutoff)) {
  argv$data_cutoff <- NULL
}
if (is.na(argv$locations)) {
  argv$locations <- NULL
} else {
  argv$locations <- unlist(strsplit(
    argv$locations,
    " "
  ))
}

## hack to make argparser slightly more system-agnostic
if (argv$params == "data/params.toml") {
  argv$params <- fs::path("data", "params.toml")
}

api_creds <- cfaepim::get_api_credentials()

fit(
  argv$report_date,
  argv$outdir,
  data_cutoff_date = argv$data_cutoff,
  locations = argv$locations,
  param_path = argv$params,
  healthdata_api_key_id = api_creds$id,
  healthdata_api_key_secret = api_creds$key,
  overwrite_params = argv$overwrite_params
)
