# numpydoc ignore=GL08
import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
import pytest
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    InfectionSeedingProcess,
    SeedInfectionsFromVec,
    SeedInfectionsViaExpGrowth,
    SeedInfectionsZeroPad,
)
from pyrenew.metaclass import DistributionalRV


def test_infection_seeding_process():
    """Check that the InfectionSeedingProcess class generates can be sampled from with all InfectionSeedMethods."""
    n_timepoints = 10

    zero_pad_model = InfectionSeedingProcess(
        "zero_pad_model",
        DistributionalRV(dist=dist.LogNormal(0, 1), name="I0"),
        SeedInfectionsZeroPad(n_timepoints),
        t_unit=1,
    )

    exp_model = InfectionSeedingProcess(
        "exp_model",
        DistributionalRV(dist=dist.LogNormal(0, 1), name="I0"),
        SeedInfectionsViaExpGrowth(
            n_timepoints, DeterministicVariable(0.5, name="rate")
        ),
        t_unit=1,
    )

    vec_model = InfectionSeedingProcess(
        "vec_model",
        DeterministicVariable(jnp.arange(n_timepoints), name="I0"),
        SeedInfectionsFromVec(n_timepoints),
        t_unit=1,
    )

    for model in [zero_pad_model, exp_model, vec_model]:
        with npro.handlers.seed(rng_seed=1):
            model.sample()

    # Check that the InfectionSeedingProcess class raises an error when the wrong type of I0 is passed
    with pytest.raises(TypeError):
        InfectionSeedingProcess(
            "vec_model",
            jnp.arange(n_timepoints),
            SeedInfectionsFromVec(n_timepoints),
            t_unit=1,
        )

    with pytest.raises(TypeError):
        InfectionSeedingProcess(
            "vec_model",
            DeterministicVariable(jnp.arange(n_timepoints), name="I0"),
            3,
            t_unit=1,
        )
