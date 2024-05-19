import numpyro as npro
import numpyro.distributions as dist
from pyrenew.latent.infection_seeding_method import (
    SeedInfectionsExponential,
    SeedInfectionsRepeat,
    SeedInfectionsZeroPad,
    SeedInfectionsZeroHstack,
)
from pyrenew.latent.infection_seeding_process import InfectionSeedingProcess
import jax.numpy as jnp
n_timepoints = 10

repeat_model = InfectionSeedingProcess(
    dist.Normal(), SeedInfectionsRepeat(), n_timepoints
)
zero_pad_model = InfectionSeedingProcess(
    dist.Normal(), SeedInfectionsZeroPad(), n_timepoints
)

zero_hstack_model = InfectionSeedingProcess(
    dist.Normal(), SeedInfectionsZeroHstack(), n_timepoints
)

exp_model = InfectionSeedingProcess(
    dist.Normal(), SeedInfectionsExponential(2), n_timepoints
)

# Works:
with npro.handlers.seed(rng_seed=n_timepoints):
    repeat_dat = repeat_model.sample()
repeat_dat

# Works:
with npro.handlers.seed(rng_seed=n_timepoints):
    exp_dat = exp_model.sample()
exp_dat


# Doesn't work:
with npro.handlers.seed(rng_seed=n_timepoints):
    zero_pad_dat = zero_pad_model.sample()
zero_pad_dat

# Works:
SeedInfectionsZeroPad().seed_infections(jnp.array([11.0]), n_timepoints)

# Works:
with npro.handlers.seed(rng_seed=n_timepoints):
    zero_hstack_dat = zero_hstack_model.sample()
zero_hstack_dat

# Works:
SeedInfectionsZeroHstack().seed_infections(jnp.array([11.0]), n_timepoints)