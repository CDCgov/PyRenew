"""This is a demo"""
import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    InfectionSeedingProcess,
    SeedInfectionsExponential,
    SeedInfectionsFromVec,
    SeedInfectionsZeroPad,
)
from pyrenew.metaclass import DistributionalRV

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
    DistributionalRV(
        dist=dist.LogNormal(loc=jnp.log(80 / 0.05), scale=1.5), name="I0"
    ),
    SeedInfectionsZeroPad(n_timepoints),
)
with npro.handlers.seed(rng_seed=rng_seed):
    zero_pad_dat = zero_pad_model.sample()
zero_pad_dat

exp_model = InfectionSeedingProcess(
    DistributionalRV(
        dist=dist.LogNormal(loc=jnp.log(80 / 0.05), scale=1.5), name="I0"
    ),
    SeedInfectionsExponential(
        n_timepoints, DeterministicVariable(0.5), t_I_pre_seed=0
    ),
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
