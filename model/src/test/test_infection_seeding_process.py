# numpydoc ignore=GL08
import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
import pytest
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    InfectionInitializationProcess,
    InitializeInfectionsExponentialGrowth,
    InitializeInfectionsFromVec,
    InitializeInfectionsZeroPad,
)
from pyrenew.metaclass import DistributionalRV


def test_infection_initialization_process():
    """Check that the InfectionInitializationProcess class generates can be sampled from with all InfectionInitializationMethods."""
    n_timepoints = 10

    zero_pad_model = InfectionInitializationProcess(
        "zero_pad_model",
        DistributionalRV(dist=dist.LogNormal(0, 1), name="I0"),
        InitializeInfectionsZeroPad(n_timepoints),
        t_unit=1,
    )

    exp_model = InfectionInitializationProcess(
        "exp_model",
        DistributionalRV(dist=dist.LogNormal(0, 1), name="I0"),
        InitializeInfectionsExponentialGrowth(
            n_timepoints, DeterministicVariable(0.5, name="rate")
        ),
        t_unit=1,
    )

    vec_model = InfectionInitializationProcess(
        "vec_model",
        DeterministicVariable(jnp.arange(n_timepoints), name="I0"),
        InitializeInfectionsFromVec(n_timepoints),
        t_unit=1,
    )

    for model in [zero_pad_model, exp_model, vec_model]:
        with npro.handlers.seed(rng_seed=1):
            model.sample()

    # Check that the InfectionInitializationProcess class raises an error when the wrong type of I0 is passed
    with pytest.raises(TypeError):
        InfectionInitializationProcess(
            "vec_model",
            jnp.arange(n_timepoints),
            InitializeInfectionsFromVec(n_timepoints),
            t_unit=1,
        )

    with pytest.raises(TypeError):
        InfectionInitializationProcess(
            "vec_model",
            DeterministicVariable(jnp.arange(n_timepoints), name="I0"),
            3,
            t_unit=1,
        )
