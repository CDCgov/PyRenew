"""
Tests for the pyrenew.datasets module
"""

import numpy.testing as testing

from pyrenew.datasets import (
    load_example_infection_admission_interval,
)


def test_infection_admission_interval():
    """Test that the infection to admission interval dataset can be properly loaded"""
    df = load_example_infection_admission_interval()
    assert len(df) > 0
    assert df.shape == (55, 2)

    testing.assert_approx_equal(df["probability_mass"].mean(), 0.01818181818, 3)
