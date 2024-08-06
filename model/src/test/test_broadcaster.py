"""
Test the broadcaster utility
"""

import jax.numpy as jnp
import numpy.testing as testing
import pytest
from pyrenew.arrayutils import repeat_until_n, tile_until_n


def test_broadcaster() -> None:
    """
    Test the PeriodicBroadcaster utility.
    """
    base_array = jnp.array([1, 2, 3])

    testing.assert_array_equal(
        tile_until_n(base_array, 10),
        jnp.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1]),
    )

    # With an offset of 2
    testing.assert_array_equal(
        tile_until_n(base_array, 10, offset=2),
        jnp.array([3, 1, 2, 3, 1, 2, 3, 1, 2, 3]),
    )

    testing.assert_array_equal(
        repeat_until_n(
            data=base_array,
            n_timepoints=10,
            offset=0,
            period_size=7,
        ),
        jnp.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2]),
    )

    # With an offset of 2
    testing.assert_array_equal(
        repeat_until_n(
            data=base_array,
            n_timepoints=10,
            offset=2,
            period_size=7,
        ),
        jnp.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
    )

    with pytest.raises(ValueError, match="The data is too short to broadcast"):
        repeat_until_n(
            data=base_array,
            n_timepoints=100,
            offset=0,
            period_size=7,
        )

    # Checking the default value of `n_timepoints`
    ans0 = repeat_until_n(base_array, period_size=10)
    ans1 = repeat_until_n(
        base_array, period_size=10, n_timepoints=10 * base_array.size
    )

    testing.assert_array_equal(ans0, ans1)

    return None
