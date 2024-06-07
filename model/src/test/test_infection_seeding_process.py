# numpydoc ignore=GL08
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


def test_infection_seeding_process():
    """Check that the InfectionSeedingProcess class generates can be sampled from with all InfectionSeedMethods."""
    n_timepoints = 10

    zero_pad_model = InfectionSeedingProcess(
        DistributionalRV(dist=dist.LogNormal(0, 1), name="I0"),
        SeedInfectionsZeroPad(n_timepoints),
    )

    exp_model = InfectionSeedingProcess(
        DistributionalRV(dist=dist.LogNormal(0, 1), name="I0"),
        SeedInfectionsExponential(
            n_timepoints, DeterministicVariable(0.5, name="rate")
        ),
    )

    vec_model = InfectionSeedingProcess(
        DeterministicVariable(jnp.arange(n_timepoints), name="I0"),
        SeedInfectionsFromVec(n_timepoints),
    )

    for model in [zero_pad_model, exp_model, vec_model]:
        with npro.handlers.seed(rng_seed=1):
            model.sample()

