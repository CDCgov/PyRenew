import numpyro as npro
import numpyro.distributions as dist
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent.infection_seeding_method import (
    SeedInfectionsExponential,
    SeedInfectionsRepeat,
    SeedInfectionsZeroPad,
)
from pyrenew.latent.infection_seeding_process import InfectionSeedingProcess

n_timepoints = 10
rng_seed = 1

repeat_model = InfectionSeedingProcess(
    dist.Normal(),
    SeedInfectionsRepeat(n_timepoints),
)

zero_pad_model = InfectionSeedingProcess(
    dist.Normal(),
    SeedInfectionsZeroPad(n_timepoints),
)

exp_model = InfectionSeedingProcess(
    dist.Normal(),
    SeedInfectionsExponential(n_timepoints, DeterministicVariable(0.5)),
)

# Works:
with npro.handlers.seed(rng_seed=rng_seed):
    repeat_dat = repeat_model.sample()
repeat_dat

# Works:
with npro.handlers.seed(rng_seed=rng_seed):
    exp_dat = exp_model.sample()
exp_dat


# Works:
with npro.handlers.seed(rng_seed=rng_seed):
    zero_pad_dat = zero_pad_model.sample()
zero_pad_dat
