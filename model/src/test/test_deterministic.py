# numpydoc ignore=GL08

import jax.numpy as jnp
import numpy.testing as testing
from pyrenew.deterministic import (
    DeterministicPMF,
    DeterministicProcess,
    DeterministicVariable,
    NullProcess,
    NullVariable,
)


def test_deterministic():
    """
    Test the DeterministicVariable, DeterministicPMF, and
    DeterministicProcess classes in the deterministic module.
    """

    var1 = DeterministicVariable(
        jnp.array(
            [
                1,
            ]
        )
    )
    var2 = DeterministicPMF(jnp.array([0.25, 0.25, 0.2, 0.3]))
    var3 = DeterministicProcess(jnp.array([1, 2, 3, 4]))
    var4 = NullVariable()
    var5 = NullProcess()

    testing.assert_array_equal(
        var1.sample()[0],
        jnp.array(
            [
                1,
            ]
        ),
    )
    testing.assert_array_equal(
        var2.sample()[0],
        jnp.array([0.25, 0.25, 0.2, 0.3]),
    )
    testing.assert_array_equal(
        var3.sample(duration=5)[0],
        jnp.array([1, 2, 3, 4, 4]),
    )

    testing.assert_array_equal(
        var3.sample(duration=3)[0],
        jnp.array(
            [
                1,
                2,
                3,
            ]
        ),
    )

    testing.assert_equal(var4.sample()[0], None)
    testing.assert_equal(var5.sample(duration=1)[0], None)
