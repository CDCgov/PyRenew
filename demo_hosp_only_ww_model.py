# numpydoc ignore=GL08
import json

import numpy as np
import numpyro
import numpyro.distributions as dist
from pyrenew.metaclass import DistributionalRV
from pyrenew.model import hosp_only_ww_model

# Load the JSON file
with open(
    "scratch/stan_data_hosp_only.json",
    "r",
) as file:
    stan_data = json.load(file)

# Pylance complains about not being able to find variables
for key in stan_data:
    print(key)
#     if isinstance(stan_data[key], (list, tuple)) and len(stan_data[key]) == 1:
#         globals()[key] = stan_data[key][0]
#     else:
#         globals()[key] = stan_data[key]

i0_over_n_prior_a = stan_data["i0_over_n_prior_a"][0]
i0_over_n_prior_b = stan_data["i0_over_n_prior_b"][0]
initial_growth_prior_mean = stan_data["initial_growth_prior_mean"][0]
initial_growth_prior_sd = stan_data["initial_growth_prior_sd"][0]
uot = stan_data["uot"][0]

r_prior_mean = stan_data["r_prior_mean"][0]
r_prior_sd = stan_data["r_prior_sd"][0]


def convert_to_logmean_log_sd(mean, sd):
    # numpydoc ignore=GL08
    logmean = np.log(
        np.power(mean, 2) / np.sqrt(np.power(sd, 2) + np.power(mean, 2))
    )
    logsd = np.sqrt(np.log(1 + (np.power(sd, 2) / np.power(mean, 2))))
    return logmean, logsd


r_logmean, r_logsd = convert_to_logmean_log_sd(r_prior_mean, r_prior_sd)


i0_over_n_rv = DistributionalRV(
    dist.Beta(i0_over_n_prior_a, i0_over_n_prior_b), "i0_over_n_rv"
)

initialization_rate_rv = DistributionalRV(
    dist.Normal(initial_growth_prior_mean, initial_growth_prior_sd), "rate"
)
state_pop = stan_data["state_pop"][0]
my_model = hosp_only_ww_model(
    state_pop=state_pop,
    i0_over_n_rv=i0_over_n_rv,
    initialization_rate_rv=initialization_rate_rv,
    n_initialization_points=uot,
)

with numpyro.handlers.seed(rng_seed=200):
    my_model_samp = my_model.sample()


my_model_samp
