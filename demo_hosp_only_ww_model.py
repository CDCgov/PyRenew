# numpydoc ignore=GL08
import json

import numpy as np
import numpyro
import numpyro.distributions as dist
from pyrenew.metaclass import DistributionalRV
from pyrenew.model import hosp_only_ww_model


def convert_to_logmean_log_sd(mean, sd):
    # numpydoc ignore=GL08
    logmean = np.log(
        np.power(mean, 2) / np.sqrt(np.power(sd, 2) + np.power(mean, 2))
    )
    logsd = np.sqrt(np.log(1 + (np.power(sd, 2) / np.power(mean, 2))))
    return logmean, logsd


# Load the JSON file
with open(
    "scratch/stan_data_hosp_only.json",
    "r",
) as file:
    stan_data = json.load(file)

# Pylance complains about not being able to find variables
# for key in stan_data:
#     print(key)
#     if isinstance(stan_data[key], (list, tuple)) and len(stan_data[key]) == 1:
#         globals()[key] = stan_data[key][0]
#     else:
#         globals()[key] = stan_data[key]

i0_over_n_prior_a = stan_data["i0_over_n_prior_a"][0]
i0_over_n_prior_b = stan_data["i0_over_n_prior_b"][0]
i0_over_n_rv = DistributionalRV(
    "i0_over_n_rv", dist.Beta(i0_over_n_prior_a, i0_over_n_prior_b)
)

initial_growth_prior_mean = stan_data["initial_growth_prior_mean"][0]
initial_growth_prior_sd = stan_data["initial_growth_prior_sd"][0]
initialization_rate_rv = DistributionalRV(
    "rate", dist.Normal(initial_growth_prior_mean, initial_growth_prior_sd)
)

r_prior_mean = stan_data["r_prior_mean"][0]
r_prior_sd = stan_data["r_prior_sd"][0]
r_logmean, r_logsd = convert_to_logmean_log_sd(r_prior_mean, r_prior_sd)
log_r_mu_intercept_rv = DistributionalRV(
    "log_r_mu_intercept_rv", dist.Normal(r_logmean, r_logsd)
)


eta_sd_sd = stan_data["eta_sd_sd"][0]
eta_sd_rv = DistributionalRV(
    "eta_sd", dist.TruncatedNormal(0, eta_sd_sd, low=0)
)

autoreg_rt_a = stan_data["autoreg_rt_a"][0]
autoreg_rt_b = stan_data["autoreg_rt_b"][0]
autoreg_rt_rv = DistributionalRV(
    "autoreg_rt", dist.Beta(autoreg_rt_a, autoreg_rt_b)
)


uot = stan_data["uot"][0]
state_pop = stan_data["state_pop"][0]

my_model = hosp_only_ww_model(
    state_pop=state_pop,
    i0_over_n_rv=i0_over_n_rv,
    initialization_rate_rv=initialization_rate_rv,
    log_r_mu_intercept_rv=log_r_mu_intercept_rv,
    autoreg_rt_rv=autoreg_rt_rv,  # ar process
    eta_sd_rv=eta_sd_rv,  # sd of random walk for ar process
    n_initialization_points=uot,
    n_timepoints=50,
)

with numpyro.handlers.seed(rng_seed=200):
    my_model_samp = my_model.sample()

my_model_samp
