"""
Test the broadcaster utility
"""

import jax.numpy as jnp
import numpy.testing as testing
import pytest
from pyrenew.arrayutils import PeriodicBroadcaster


def test_broadcaster() -> None:
    """
    Test the PeriodicBroadcaster utility.
    """
    base_array = jnp.array([1, 2, 3])

    with pytest.warns(
        UserWarning,
        match="Period size is not used when broadcasting with the "
        "'tile' method.",
    ):
        tile_broadcaster = PeriodicBroadcaster(
            offset=0, period_size=7, broadcast_type="tile"
        )

    testing.assert_array_equal(
        tile_broadcaster(base_array, 10),
        jnp.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1]),
    )

    repeat_broadcaster = PeriodicBroadcaster(
        offset=0, period_size=7, broadcast_type="repeat"
    )

    testing.assert_array_equal(
        repeat_broadcaster(base_array, 10),
        jnp.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2]),
    )

    with pytest.raises(ValueError, match="The data is too short to broadcast"):
        repeat_broadcaster(base_array, 100)

    return None
