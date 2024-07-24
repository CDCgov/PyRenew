# numpydoc ignore=GL08
import numpyro
import numpyro.distributions as dist
from pyrenew.metaclass import DistributionalRV
from pyrenew.model import hosp_only_ww_model

n_initialization_points = 50
i0_over_n_prior_a = 1
i0_over_n_prior_b = 1
i0_over_n_rv = DistributionalRV(
    dist.Beta(i0_over_n_prior_a, i0_over_n_prior_b), "i0_over_n_rv"
)

initialization_rate_rv = DistributionalRV(dist.Normal(0, 0.01), "rate")
state_pop = 1000
my_model = hosp_only_ww_model(
    state_pop=state_pop,
    i0_over_n_rv=i0_over_n_rv,
    initialization_rate_rv=initialization_rate_rv,
    n_initialization_points=n_initialization_points,
)

with numpyro.handlers.seed(rng_seed=200):
    my_model_samp = my_model.sample()


my_model_samp
