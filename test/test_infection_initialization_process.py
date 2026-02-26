# numpydoc ignore=GL08
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    InfectionInitializationProcess,
    InitializeInfectionsExponentialGrowth,
    InitializeInfectionsFromVec,
    InitializeInfectionsZeroPad,
)
from pyrenew.randomvariable import DistributionalVariable


def test_infection_initialization_process():
    """Check that the InfectionInitializationProcess class generates can be sampled from with all InfectionInitializationMethods."""
    n_timepoints = 10

    zero_pad_model = InfectionInitializationProcess(
        "zero_pad_model",
        DistributionalVariable(name="I0", distribution=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints),
    )

    exp_model = InfectionInitializationProcess(
        "exp_model",
        DistributionalVariable(name="I0", distribution=dist.LogNormal(0, 1)),
        InitializeInfectionsExponentialGrowth(
            n_timepoints, DeterministicVariable(name="rate", value=0.5)
        ),
    )

    vec_model = InfectionInitializationProcess(
        "vec_model",
        DeterministicVariable(name="I0", value=jnp.arange(n_timepoints)),
        InitializeInfectionsFromVec(n_timepoints),
    )

    with numpyro.handlers.seed(rng_seed=1):
        zero_pad_result = zero_pad_model()
        exp_result = exp_model()
        vec_result = vec_model()

    # All results should have shape (n_timepoints,)
    assert zero_pad_result.shape == (n_timepoints,)
    assert exp_result.shape == (n_timepoints,)
    assert vec_result.shape == (n_timepoints,)

    # Zero-pad: all but last element should be zero
    assert jnp.all(zero_pad_result[:-1] == 0)

    # Exponential growth: all values should be positive (LogNormal I0)
    assert jnp.all(exp_result > 0)

    # Vec (identity passthrough): should equal jnp.arange(n_timepoints)
    assert jnp.array_equal(vec_result, jnp.arange(n_timepoints))

    # Check that the InfectionInitializationProcess class raises an error when the wrong type of I0 is passed
    with pytest.raises(TypeError):
        InfectionInitializationProcess(
            "vec_model",
            jnp.arange(n_timepoints),
            InitializeInfectionsFromVec(n_timepoints),
        )

    with pytest.raises(TypeError):
        InfectionInitializationProcess(
            "vec_model",
            DeterministicVariable(name="I0", value=jnp.arange(n_timepoints)),
            3,
        )
