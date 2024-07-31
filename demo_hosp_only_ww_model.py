# numpydoc ignore=GL08
import json

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as transforms
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import DistributionalRV, TransformedRandomVariable
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
    "rate",
    dist.TruncatedNormal(
        loc=initial_growth_prior_mean,
        scale=initial_growth_prior_sd,
        low=-1,
        high=1,
    ),
)
# Can reasonably switch to non-Truncated

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


generation_interval_pmf_rv = DeterministicVariable(
    "generation_interval_pmf", jnp.array(stan_data["generation_interval"])
)
# THIS MIGHT HAVE TO BE REVERSED
infection_feedback_pmf_rv = DeterministicVariable(
    "infection_feedback_pmf", jnp.array(stan_data["infection_feedback_pmf"])
)

# infection_feedback ~ lognormal(inf_feedback_prior_logmean, inf_feedback_prior_logsd);
inf_feedback_prior_logmean = stan_data["inf_feedback_prior_logmean"][0]
inf_feedback_prior_logsd = stan_data["inf_feedback_prior_logsd"][0]
inf_feedback_strength_rv = TransformedRandomVariable(
    "inf_feedback",
    DistributionalRV(
        "inf_feedback_raw",
        dist.LogNormal(inf_feedback_prior_logmean, inf_feedback_prior_logsd),
    ),
    transforms=transforms.AffineTransform(loc=0, scale=-1),
)
# Could be reparameterized?

p_hosp_prior_mean = stan_data["p_hosp_prior_mean"][0]
p_hosp_sd_logit = stan_data["p_hosp_sd_logit"][0]

p_hosp_mean_rv = DistributionalRV(
    "p_hosp_mean",
    dist.Normal(transforms.logit(p_hosp_prior_mean), p_hosp_sd_logit),
)  # logit scale

p_hosp_w_sd_sd = stan_data["p_hosp_w_sd_sd"][0]
p_hosp_w_sd_rv = DistributionalRV(
    "p_hosp_w_sd_sd", dist.TruncatedNormal(0, p_hosp_w_sd_sd, low=0)
)

autoreg_p_hosp_a = stan_data["autoreg_p_hosp_a"][0]
autoreg_p_hosp_b = stan_data["autoreg_p_hosp_b"][0]
autoreg_p_hosp_rv = DistributionalRV(
    "autoreg_p_hosp", dist.Beta(autoreg_p_hosp_a, autoreg_p_hosp_b)
)

# p_hosp_w ~ std_normal();


uot = stan_data["uot"][0]
state_pop = stan_data["state_pop"][0]

my_model = hosp_only_ww_model(
    state_pop=state_pop,
    i0_over_n_rv=i0_over_n_rv,
    initialization_rate_rv=initialization_rate_rv,
    log_r_mu_intercept_rv=log_r_mu_intercept_rv,
    autoreg_rt_rv=autoreg_rt_rv,  # ar process
    eta_sd_rv=eta_sd_rv,  # sd of random walk for ar process,
    generation_interval_pmf_rv=generation_interval_pmf_rv,
    infection_feedback_pmf_rv=infection_feedback_pmf_rv,
    infection_feedback_strength_rv=inf_feedback_strength_rv,
    p_hosp_mean_rv=p_hosp_mean_rv,
    p_hosp_w_sd_rv=p_hosp_w_sd_rv,
    autoreg_p_hosp_rv=autoreg_p_hosp_rv,
    n_initialization_points=uot,
)

with numpyro.handlers.seed(rng_seed=202):
    my_model_samp = my_model.sample(n_timepoints=50)
