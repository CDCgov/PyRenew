# numpydoc ignore=GL08

import numpyro.distributions as dist
import pytest

from pyrenew.metaclass import DistributionalRV, RandomVariable, _assert_type


def test_valid_assertion_types():
    """
    Test valid assertion types in _assert_type.
    """

    values = [
        5,
        "Hello",
        (1,),
        DistributionalRV(name="rv", distribution=dist.Beta(1, 1)),
    ]
    arg_names = ["input_int", "input_string", "input_tuple", "input_rv"]
    input_types = [int, str, tuple, RandomVariable]

    for arg, value, input in zip(arg_names, values, input_types):
        _assert_type(arg, value, input)


def test_invalid_assertion_types():
    """
    Test invalid assertion types in _assert_type.
    """

    values = [None] * 4
    arg_names = ["input_int", "input_string", "input_tuple", "input_rv"]
    input_types = [int, str, tuple, RandomVariable]

    for arg, value, input in zip(arg_names, values, input_types):
        with pytest.raises(TypeError):
            _assert_type(arg, value, input)
