#' Augment a set of epidemia data with a
#' preobservation period
#'
#' This allows for more robust initialization
#' of the renewal process.
#'
#' @param data The data to augment
#' @param n_pre_observation_days number of
#' pre-observation days with which to augment it.
#' @return the augmented data frame, as a [tibble::tibble()]
#' @export
add_pre_observation_period <- function(data,
                                       n_pre_observation_days) {
  pre_observation_data <- tibble::tibble(
    date = min(data$date) - n_pre_observation_days:1,
    hosp = NA,
    nonobservation_period = TRUE,
    location = data$location[1],
    population = data$population[1],
    day_of_week = factor(
      lubridate::wday(date,
        label = TRUE
      ),
      ordered = FALSE,
      levels = levels(data$day_of_week)
    ),
    is_holiday = FALSE,
    is_post_holiday = FALSE,
    recency = 0,
    week = data$week[1] ## this prevents RW during seeding
  )

  aug_data <- pre_observation_data |>
    dplyr::bind_rows(data) |>
    dplyr::mutate(hosp = ifelse(nonobservation_period,
      0,
      hosp
    ))
  print(aug_data |> tail())
  return(aug_data)
}

#' Build the R(t) component of a "covariate
#' light" epidemia model.
#'
#' @param rt_intercept_prior_mode prior mode
#' for R(t) intercept. Prior is Normal on the
#' transformed (scaled logit) scale.
#' @param rt_intercept_prior_scale prior standard
#' deviation (scale) for R(t) intercept. Prior is
#' Normal on the transformed (scaled logit) scale.
#' @param max_rt Maximum permitted R(t) value (upper
#' limit of the scaled logit), on the natural scale.
#' R(t) will be permitted to range between 0 and
#' this value.
#' @param rw_prior_scale prior standard deviation
#' (scale parameter) for the random walk on R(t).
#' Random walk steps are Normal on the transformed
#' (scaled logit) scale.
#' @return the R(t) model, as the output
#' of an [epidemia::epirt()] call.
#' @export
build_light_rt <- function(rt_intercept_prior_mode,
                           rt_intercept_prior_scale,
                           max_rt,
                           rw_prior_scale) {
  rt_model <- epidemia::epirt(
    formula = as.formula(
      sprintf(
        paste0(
          "R(location, date) ~ 1 + ",
          "rw(time = week, gr = location, prior_scale = %f)"
        ),
        rw_prior_scale
      )
    ),
    prior_intercept = rstanarm::normal(
      location = rt_intercept_prior_mode,
      scale = rt_intercept_prior_scale
    ),
    link = epidemia::scaled_logit(K = max_rt)
  )

  return(rt_model)
}

#' Build the observation component of a "covariate
#' light" epidemia model.
#'
#' @param inf_to_hosp_dist infection to hospitalization
#' delay distribution, passed as the `i2o` parameter
#' to [epidemia::epiobs()].
#' @param ihr_intercept_prior_mode Normal prior mode for
#' the overall infection (observed) hospitalization rate
#' (i.e. the probability that an arbitrary infected individual
#' gets observed admitted to the hospital), before taking
#' into account any covariates.
#' Specified on the transformed scale (see `link` parameter).
#' @param ihr_intercept_prior_scale Normal prior scale for the
#' intercept of the regression predicting the
#' infection (observed) hospitalization rate
#' (i.e. the probability that an arbitrary infected individual
#' gets observed admitted to the hospital), before taking
#' into account any covariates.
#' Specified on the transformed scale (see `link` parameter).
#' @param day_of_week_eff_prior_modes Normal prior
#' modes for the day of the week effect on observation
#' probability, relative to the reference day of the week.
#' Should be a vector of length 6.
#' Specified on the transformed scale (see `link` parameter).
#' @param day_of_week_eff_prior_scales Normal prior
#' scales for the day of the week effect on observation
#' probability, relative to the reference day of the week.
#' Should be a vector of length 6.
#' Specified on the transformed scale (see `link` parameter).
#' @param non_obs_effect_prior_mode Normal prior
#' mode for the change in the observation probability
#' during the nonobservation (seeding) period. Useful
#' for model initialization. Should typically be a large
#' negative number.
#' Specified on the transformed scale (see `link` parameter).
#' @param non_obs_effect_prior_scale Normal prior
#' scalefor the change in the observation probability
#' during the nonobservation (seeding) period. Useful
#' for model initialization. Should typically be a small
#' number, to enforce the large negative effect given in
#' non_obs_effect_prior_mode.
#' Specified on the transformed scale (see `link` parameter).
#' @param inv_dispersion_prior_mode Normal prior mode
#' for the reciprocal dispersion of the negative binomial
#' observation process.
#' @param inv_dispersion_prior_mode Normal prior scale
#' for the reciprocal dispersion of the negative binomial
#' observation process.
#' @param link link function for the observation model,
#' passed as the `link` parameter to [epidemia::epiobs()]
#' Default `"logit"`.
#' @return the observation model, as the output
#' of an [epidemia::epiobs()] call.
#' @export
build_light_obs <- function(inf_to_hosp_dist,
                            ihr_intercept_prior_mode,
                            ihr_intercept_prior_scale,
                            day_of_week_eff_prior_modes,
                            day_of_week_eff_prior_scales,
                            holiday_eff_prior_mode,
                            holiday_eff_prior_scale,
                            post_holiday_eff_prior_mode,
                            post_holiday_eff_prior_scale,
                            non_obs_effect_prior_mode,
                            non_obs_effect_prior_scale,
                            inv_dispersion_prior_mode,
                            inv_dispersion_prior_scale,
                            link = "logit") {
  return(epidemia::epiobs(
    formula = as.formula(paste0(
      "hosp ~ 1 + day_of_week + ",
      "is_holiday + ",
      "is_post_holiday + ",
      "nonobservation_period"
    )),
    ## Add a covariate for the
    ## nonobservation window to
    ## leave an initial evolution
    ## period with no observations
    i2o = inf_to_hosp_dist,
    link = link,
    family = "neg_binom",
    prior_intercept = rstanarm::normal(
      location = ihr_intercept_prior_mode,
      scale = ihr_intercept_prior_scale
    ),
    prior = rstanarm::normal(
      location = c(
        day_of_week_eff_prior_modes,
        holiday_eff_prior_mode,
        post_holiday_eff_prior_mode,
        non_obs_effect_prior_mode
      ),
      ## a large negative non_obs_effect
      ## effectively conditions on detection
      ## prob = 0 outside the observation period
      scale = c(
        day_of_week_eff_prior_scales,
        holiday_eff_prior_scale,
        post_holiday_eff_prior_scale,
        non_obs_effect_prior_scale
        ## non-obs prior scale
        ## should be small to
        ## enforce non-obs effect
        ## close to (large negative) mode
      )
    ),
    prior_aux = rstanarm::normal(
      location = inv_dispersion_prior_mode,
      scale = inv_dispersion_prior_scale
    )
  ))
}

#' Build a complete "covariate light" epidemia
#' model
#'
#' For the given state with
#' the given parameter list
#'
#' @param state state for which to build the model
#' @param clean_data data frame of all fitting data
#' @param params the parameter list
#' @param n_warmup number of warmup samples for Stan to draw per chain.
#' Default `1000`.
#' @param n_iter total number of iterations for Stan per chain.
#' Default `2000`.
#' @param n_chains number of separate NUTS chains to run.
#' Default `4`.
#' @param max_treedepth maximum treedepth for NUTS,
#' passed to [rstan::sampling()]. Default `11`.
#' @param adapt_delta target acceptance probability
#' for NUTS adaptation, passed to [rstan::sampling()].
#' Default `0.85`.
#' @param refresh How often to print Stan progress
#' to terminal. Default `0` (never).
#' @return a list of arguments that can be passed
#' to [epidemia::epim()]
#' @export
build_state_light_model <- function(
    state,
    clean_data,
    params,
    n_warmup = 1000,
    n_iter = 2000,
    n_chains = 4,
    max_treedepth = 11,
    adapt_delta = 0.85,
    refresh = 0) {
  rw_prior_scale <- params$weekly_rw_prior_scale
  rt_model <- build_light_rt(
    params$rt_intercept_prior_mode,
    params$rt_intercept_prior_scale,
    params$max_rt,
    params$weekly_rw_prior_scale
  )
  ## make sure day_of_week is properly
  ## set up as a factor
  dow_levels <- levels(
    lubridate::wday("2023-01-01",
      label = TRUE,
      week_start = params$reference_day_of_week
    )
  )
  clean_data <- clean_data |>
    dplyr::mutate(
      day_of_week = factor(day_of_week,
        ordered = FALSE,
        levels = dow_levels
      )
    )
  # create the mean infections per day to use in prior_seeds
  # to population adjust the seeded infections
  mode_ihr <- plogis(params$ihr_intercept_prior_mode)
  state_data <- clean_data |>
    dplyr::filter(location == !!state)
  mean_inf_df <- state_data |>
    dplyr::distinct(location, population, first_week_hosp) |>
    dplyr::mutate(
      mean_seed_inf_per_day = (
        (params$inf_model_prior_infections_per_capita * population) +
          (first_week_hosp / (!!mode_ihr * 7))
      )
    )

  mean_inf_val <- mean_inf_df$mean_seed_inf_per_day[1]
  infection_model <- epidemia::epiinf(
    params$generation_time_dist,
    seed_days = params$inf_model_seeds,
    prior_seeds = rstanarm::exponential(1 / mean_inf_val),
    pop_adjust = TRUE,
    pops = "population",
    prior_susc = rstanarm::normal(
      location = params$susceptible_fraction_prior_mode,
      scale = params$susceptible_fraction_prior_scale
    )
  )
  obs_model <- build_light_obs(
    params$inf_to_hosp_dist,
    params$ihr_intercept_prior_mode,
    params$ihr_intercept_prior_scale,
    params$day_of_week_effect_prior_modes,
    params$day_of_week_effect_prior_scales,
    params$holiday_eff_prior_mode,
    params$holiday_eff_prior_scale,
    params$post_holiday_eff_prior_mode,
    params$post_holiday_eff_prior_scale,
    params$non_obs_effect_prior_mode,
    params$non_obs_effect_prior_scale,
    params$reciprocal_dispersion_prior_mode,
    params$reciprocal_dispersion_prior_scale
  )
  stan_data <- state_data |>
    dplyr::select(
      date,
      location,
      population,
      hosp,
      day_of_week,
      is_holiday,
      is_post_holiday,
      recency,
      nonobservation_period,
      week
    )
  aug_stan_data <- add_pre_observation_period(
    stan_data,
    params$n_pre_observation_days
  )
  epim_args <- list(
    rt = rt_model,
    obs = obs_model,
    inf = infection_model,
    data = aug_stan_data,
    warmup = n_warmup,
    iter = n_iter,
    seed = params$seed,
    chains = n_chains,
    refresh = refresh,
    control = list(
      max_treedepth = max_treedepth,
      adapt_delta = adapt_delta
    )
  )
  return(epim_args)
}
