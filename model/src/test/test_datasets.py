import numpy.testing as testing
from pyrenew.datasets import load_wastewater


def test_loading_wastewater():
    """Test that the wastewater dataset can be properly loaded"""
    df = load_wastewater()
    assert len(df) > 0
    assert df.shape == (635, 14)

    testing.assert_approx_equal(df["daily_hosp_admits"].mean(), 12.8888, 3)
    testing.assert_approx_equal(df["lod_sewage"].mean(), 3.841025, 3)
