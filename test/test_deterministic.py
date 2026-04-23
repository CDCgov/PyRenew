# numpydoc ignore=GL08


import jax.numpy as jnp
import numpy.testing as testing

from pyrenew.deterministic import (
    DeterministicPMF,
    DeterministicVariable,
    NullObservation,
    NullVariable,
)


def test_deterministic():
    """
    Test the DeterministicVariable and DeterministicPMF classes in the
    deterministic module.
    """

    var1 = DeterministicVariable(
        name="var1",
        value=jnp.array(
            [
                1,
            ]
        ),
    )
    var2 = DeterministicPMF(name="var2", value=jnp.array([0.25, 0.25, 0.2, 0.3]))
    var3 = NullVariable()

    testing.assert_array_equal(
        var1(),
        jnp.array(
            [
                1,
            ]
        ),
    )
    testing.assert_array_equal(
        var2(),
        jnp.array([0.25, 0.25, 0.2, 0.3]),
    )

    testing.assert_equal(var3(), None)


def test_null_observation():
    """
    Test that NullObservation can be constructed and
    that its sample method returns None.
    """
    null_obs = NullObservation()
    assert null_obs.sample(mu=jnp.array([1.0, 2.0]), obs=None) is None
    assert null_obs.sample(mu=jnp.array([1.0]), obs=jnp.array([1.0])) is None


def test_deterministic_pmf_size():
    """
    Test that DeterministicPMF.size() returns the correct size.
    """
    pmf = DeterministicPMF(name="test_pmf", value=jnp.array([0.25, 0.25, 0.2, 0.3]))
    assert pmf.size() == 4
