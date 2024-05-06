import numpy.testing as testing
from pyrenew.datasets import (
    load_generation_interval,
    load_infection_admission_interval,
    load_wastewater,
)


def test_loading_wastewater():
    """Test that the wastewater dataset can be properly loaded"""
    df = load_wastewater()
    assert len(df) > 0
    assert df.shape == (635, 14)

    testing.assert_approx_equal(df["daily_hosp_admits"].mean(), 12.8888, 3)
    testing.assert_approx_equal(df["load_sewage"].mean(), 3.841025, 3)


def test_gen_int():
    """Test that the generation interval dataset can be properly loaded"""
    df = load_generation_interval()
    assert len(df) > 0
    assert df.shape == (15, 2)

    testing.assert_approx_equal(df["probability_mass"].mean(), 0.0666666, 3)


def test_infection_admission_interval():
    """Test that the infection to admission interval dataset can be properly loaded"""
    df = load_infection_admission_interval()
    assert len(df) > 0
    assert df.shape == (55, 2)

    testing.assert_approx_equal(
        df["probability_mass"].mean(), 0.01818181818, 3
    )
