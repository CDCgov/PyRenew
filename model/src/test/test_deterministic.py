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
        name="var1",
        vars=jnp.array(
            [
                1,
            ]
        ),
    )
    var2 = DeterministicPMF(
        name="var2", vars=jnp.array([0.25, 0.25, 0.2, 0.3])
    )
    var3 = DeterministicProcess(name="var3", vars=jnp.array([1, 2, 3, 4]))
    var4 = NullVariable()
    var5 = NullProcess()

    testing.assert_array_equal(
        var1()[0],
        jnp.array(
            [
                1,
            ]
        ),
    )
    testing.assert_array_equal(
        var2()[0],
        jnp.array([0.25, 0.25, 0.2, 0.3]),
    )
    testing.assert_array_equal(
        var3(duration=5)[0],
        jnp.array([1, 2, 3, 4, 4]),
    )

    testing.assert_array_equal(
        var3(duration=3)[0],
        jnp.array(
            [
                1,
                2,
                3,
            ]
        ),
    )

    testing.assert_equal(var4()[0], None)
    testing.assert_equal(var5(duration=1)[0], None)
