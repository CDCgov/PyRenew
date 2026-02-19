"""
Utility functions for processing arrays.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike


class PeriodicProcessSample(NamedTuple):
    """
    A container for holding the output from `process.PeriodicProcess()`.

    Attributes
    ----------
    value
        The sampled quantity.
    """

    value: ArrayLike | None = None

    def __repr__(self) -> str:
        return f"PeriodicProcessSample(value={self})"


def tile_until_n(
    data: ArrayLike,
    n_timepoints: int,
    offset: int = 0,
) -> ArrayLike:
    """
    Tile the data until it reaches `n_timepoints`.

    Parameters
    ----------
    data
        Data to broadcast.
    n_timepoints
        Duration of the sequence.
    offset
        Relative point at which data starts, must be a non-negative integer.
        Default is zero, i.e., no offset.

    Notes
    -----
    Using the `offset` parameter, the function will start the broadcast
    from the `offset`-th element of the data. If the data is shorter than
    `n_timepoints`, the function will tile the data until it
    reaches `n_timepoints`.

    Returns
    -------
    ArrayLike
        Tiled data.
    """

    # Data starts should be a positive integer
    assert isinstance(offset, int), (
        f"offset should be an integer. It is {type(offset)}."
    )

    assert 0 <= offset, f"offset should be a positive integer. It is {offset}."

    return jnp.tile(data, (n_timepoints // data.size) + 1)[
        offset : (offset + n_timepoints)
    ]


def repeat_until_n(
    data: ArrayLike,
    period_size: int,
    n_timepoints: int,
    offset: int = 0,
) -> ArrayLike:
    """
    Repeat each entry in `data` a given number of times (`period_size`)
    until an array of length `n_timepoints` has been produced.

    Notes
    -----
    Using the `offset` parameter, the function will offset the data after
    the repeat operation. So, if the offset is 2, the repeat operation
    will repeat the data until `n_timepoints + 2` and then offset the
    data by 2, returning an array of size `n_timepoints`. This is a way to start
    part-way into a period. For example, the following code will each array
    element four times until 10 timepoints and then offset the data by 2:

    .. code-block:: python
      data = jnp.array([1, 2, 3])
      repeat_until_n(data, 4, 10, 2)
      # Array([1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=int32)

    Parameters
    ----------
    data
        Data to broadcast.
    period_size
        Size of the period for the repeat broadcast.
    n_timepoints
        Duration of the sequence.
    offset
        Relative point at which data starts, must be between 0 and
        period_size - 1. By default 0, i.e., no offset.

    Returns
    -------
    ArrayLike
        Repeated data.
    """

    # Data starts should be a positive integer
    assert isinstance(offset, int), (
        f"offset should be an integer. It is {type(offset)}."
    )

    assert 0 <= offset, f"offset should be a positive integer. It is {offset}."

    # Period size should be a positive integer
    assert isinstance(period_size, int), (
        f"period_size should be an integer. It is {type(period_size)}."
    )

    assert period_size > 0, (
        f"period_size should be a positive integer. It is {period_size}."
    )

    assert offset <= period_size - 1, (
        "offset should be less than or equal to period_size - 1."
        f"It is {offset}. It should be less than or equal "
        f"to {period_size - 1}."
    )

    if (data.size * period_size) < (n_timepoints + offset):
        raise ValueError(
            "The data is too short to broadcast to the given number "
            f"of timepoints + offset ({n_timepoints + offset}). The "
            "repeated data would have a size of data.size * "
            f"period_size = {data.size} * {period_size} = "
            f"{data.size * period_size}."
        )

    return jnp.repeat(
        a=data,
        repeats=period_size,
        total_repeat_length=n_timepoints + offset,
    )[offset : (offset + n_timepoints)]
