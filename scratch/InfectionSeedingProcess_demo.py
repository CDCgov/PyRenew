import jax.numpy as jnp
import numpyro as npro
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import Infections0
from pyrenew.latent.infection_seeding_method import (
    SeedInfectionsExponential,
    SeedInfectionsFromVec,
    SeedInfectionsZeroPad,
)
from pyrenew.latent.infection_seeding_process import InfectionSeedingProcess

n_timepoints = 10
rng_seed = 1

I0 = jnp.array([4])
I0_long = jnp.arange(10)

# Testing SeedInfections functions with __call__ method
SeedInfectionsZeroPad(n_timepoints).seed_infections(I0)
SeedInfectionsExponential(
    n_timepoints, rate=DeterministicVariable(0.5)
).seed_infections(I0)
SeedInfectionsFromVec(n_timepoints).seed_infections(I0_long)

# Testing SeedInfections functions within InfectionSeedingProcess
zero_pad_model = InfectionSeedingProcess(
    Infections0(),
    SeedInfectionsZeroPad(n_timepoints),
)
with npro.handlers.seed(rng_seed=rng_seed):
    zero_pad_dat = zero_pad_model.sample()
zero_pad_dat

exp_model = InfectionSeedingProcess(
    Infections0(),
    SeedInfectionsExponential(n_timepoints, DeterministicVariable(0.5)),
)
with npro.handlers.seed(rng_seed=rng_seed):
    exp_dat = exp_model.sample()
exp_dat

vec_model = InfectionSeedingProcess(
    DeterministicVariable(I0_long), SeedInfectionsFromVec(n_timepoints)
)
with npro.handlers.seed(rng_seed=rng_seed):
    vec_dat = vec_model.sample()
vec_dat
