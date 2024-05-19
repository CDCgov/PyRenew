import numpyro as npro
import numpyro.distributions as dist
from pyrenew.latent.infection_seeding_method import (
    SeedInfectionsExponential,
    SeedInfectionsRepeat,
    SeedInfectionsZeroPad,
)
from pyrenew.latent.infection_seeding_process import InfectionSeedingProcess

repeat_model = InfectionSeedingProcess(
    dist.Normal(), SeedInfectionsRepeat(), 10
)
zero_model = InfectionSeedingProcess(
    dist.Normal(), SeedInfectionsZeroPad(), 10
)
exp_model = InfectionSeedingProcess(
    dist.Normal(), SeedInfectionsExponential(2), 10
)

with npro.handlers.seed(rng_seed=10):
    repeat_dat = repeat_model.sample()
repeat_dat

with npro.handlers.seed(rng_seed=10):
    zero_dat = zero_model.sample()
zero_dat


with npro.handlers.seed(rng_seed=10):
    exp_dat = exp_model.sample()
exp_dat
