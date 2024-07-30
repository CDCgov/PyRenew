# numpydoc ignore=GL08

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import pytest
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
        value=jnp.array(
            [
                1,
            ]
        ),
    )
    var2 = DeterministicPMF(
        name="var2", value=jnp.array([0.25, 0.25, 0.2, 0.3])
    )
    var3 = DeterministicProcess(name="var3", value=jnp.array([1, 2, 3, 4]))
    var4 = NullVariable()
    var5 = NullProcess()

    testing.assert_array_equal(
        var1()[0].value,
        jnp.array(
            [
                1,
            ]
        ),
    )
    testing.assert_array_equal(
        var2()[0].value,
        jnp.array([0.25, 0.25, 0.2, 0.3]),
    )
    testing.assert_array_equal(
        var3(duration=5)[0].value,
        jnp.array([1, 2, 3, 4, 4]),
    )

    testing.assert_array_equal(
        var3(duration=3)[0].value,
        jnp.array(
            [
                1,
                2,
                3,
            ]
        ),
    )

    testing.assert_equal(var4()[0].value, None)
    testing.assert_equal(var5(duration=1)[0].value, None)


def test_deterministic_validation():
    """
    Check that validation methods for DeterministicVariable
    work as expected.
    """
    # validation should fail on construction
    some_non_array_likes = [
        {"a": jnp.array([1, 2.5, 3])},
        # a valid pytree, but not an arraylike
        "a string",
    ]
    some_array_likes = [
        5,
        -3.023523,
        np.array([1, 3.32, 5]),
        jnp.array([-32, 23]),
        jnp.array(-32),
        np.array(5),
    ]

    for non_arraylike_val in some_non_array_likes:
        with pytest.raises(
            ValueError, match="value is not an ArrayLike object"
        ):
            # the class's validation function itself
            # should raise an error when passed a
            # non arraylike value
            DeterministicVariable.validate(non_arraylike_val)

        with pytest.raises(
            ValueError, match="value is not an ArrayLike object"
        ):
            # validation should fail on constructor call
            DeterministicVariable(
                value=non_arraylike_val, name="invalid_variable"
            )

    # validation should succeed with ArrayLike
    for arraylike_val in some_array_likes:
        DeterministicVariable.validate(arraylike_val)
        DeterministicVariable(value=arraylike_val, name="valid_variable")
