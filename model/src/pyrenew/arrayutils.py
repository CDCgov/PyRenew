"""
Utility functions for processing arrays.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike


def pad_to_match(
    x: ArrayLike,
    y: ArrayLike,
    fill_value: float = 0.0,
    pad_direction: str = "end",
    fix_y: bool = False,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Pad the shorter array at the start or end to match the length of the longer array.

    Parameters
    ----------
    x : ArrayLike
        First array.
    y : ArrayLike
        Second array.
    fill_value : float, optional
        Value to use for padding, by default 0.0.
    pad_direction : str, optional
        Direction to pad the shorter array, either "start" or "end", by default "end".
    fix_y : bool, optional
        If True, raise an error when `y` is shorter than `x`, by default False.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        Tuple of the two arrays with the same length.
    """
    x = jnp.atleast_1d(x)
    y = jnp.atleast_1d(y)
    x_len = x.size
    y_len = y.size
    pad_size = abs(x_len - y_len)

    pad_width = {"start": (pad_size, 0), "end": (0, pad_size)}.get(
        pad_direction, None
    )

    if pad_width is None:
        raise ValueError(
            "pad_direction must be either 'start' or 'end'."
            f" Got {pad_direction}."
        )

    if x_len > y_len:
        if fix_y:
            raise ValueError(
                "Cannot fix y when x is longer than y."
                f" x_len: {x_len}, y_len: {y_len}."
            )
        y = jnp.pad(y, pad_width, constant_values=fill_value)

    elif y_len > x_len:
        x = jnp.pad(x, pad_width, constant_values=fill_value)

    return x, y


def pad_x_to_match_y(
    x: ArrayLike,
    y: ArrayLike,
    fill_value: float = 0.0,
    pad_direction: str = "end",
) -> ArrayLike:
    """
    Pad the `x` array at the start or end to match the length of the `y` array.

    Parameters
    ----------
    x : ArrayLike
        First array.
    y : ArrayLike
        Second array.
    fill_value : float, optional
        Value to use for padding, by default 0.0.
    pad_direction : str, optional
        Direction to pad the shorter array, either "start" or "end", by default "end".

    Returns
    -------
    Array
        Padded array.
    """
    return pad_to_match(
        x, y, fill_value=fill_value, pad_direction=pad_direction, fix_y=True
    )[0]


class PeriodicProcessSample(NamedTuple):
    """
    A container for holding the output from `process.PeriodicProcess()`.

    Attributes
    ----------
    value : ArrayLike
        The sampled quantity.
    """

    value: ArrayLike | None = None

    def __repr__(self):
        return f"PeriodicProcessSample(value={self.value})"


def tile_until_n(
    data: ArrayLike,
    n_timepoints: int,
    offset: int = 0,
) -> ArrayLike:
    """
    Tile the data until it reaches `n_timepoints`.

    Parameters
    ----------
    data : ArrayLike
        Data to broadcast.
    n_timepoints : int
        Duration of the sequence.
    offset : int
        Relative point at which data starts, must be a non-negative integer.
        Default is zero, i.e., no offset.

    Notes
    -----
    Using the `offset` parameter, the function will start the broadcast
    from the `offset`-th element of the data. If the data is shorter than
    `n_timepoints`, the function will repeat or tile the data until it
    reaches `n_timepoints`.

    Returns
    -------
    ArrayLike
        Tiled data.
    """

    # Data starts should be a positive integer
    assert isinstance(
        offset, int
    ), f"offset should be an integer. It is {type(offset)}."

    assert 0 <= offset, f"offset should be a positive integer. It is {offset}."

    return jnp.tile(data, (n_timepoints // data.size) + 1)[
        offset : (offset + n_timepoints)
    ]


def repeat_until_n(
    data: ArrayLike,
    n_timepoints: int,
    period_size: int,
    offset: int = 0,
):
    """
    Repeat the data until it reaches `n_timepoints`.

    Notes
    -----
    Using the `offset` parameter, the function will start the broadcast
    from the `offset`-th element of the data. If the data is shorter than
    `n_timepoints`, the function will repeat or tile the data until it
    reaches `n_timepoints`.

    Parameters
    ----------
    data : ArrayLike
        Data to broadcast.
    n_timepoints : int
        Duration of the sequence.
    period_size : int
        Size of the period for the repeat broadcast.
    offset : int
        Relative point at which data starts, must be between 0 and
        period_size - 1. By default 0, i.e., no offset.

    Returns
    -------
    ArrayLike
        Repeated data.
    """

    # Data starts should be a positive integer
    assert isinstance(
        offset, int
    ), f"offset should be an integer. It is {type(offset)}."

    assert 0 <= offset, f"offset should be a positive integer. It is {offset}."

    # Period size should be a positive integer
    assert isinstance(
        period_size, int
    ), f"period_size should be an integer. It is {type(period_size)}."

    assert (
        period_size > 0
    ), f"period_size should be a positive integer. It is {period_size}."

    assert offset <= period_size - 1, (
        "offset should be less than or equal to period_size - 1."
        f"It is {offset}. It should be less than or equal "
        f"to {period_size - 1}."
    )

    if (data.size * period_size) < n_timepoints:
        raise ValueError(
            "The data is too short to broadcast to "
            f"the given number of timepoints ({n_timepoints}). The "
            "repeated data would have a size of data.size * "
            f"period_size = {data.size} * {period_size} = "
            f"{data.size * period_size}."
        )

    return jnp.repeat(data, period_size)[offset : (offset + n_timepoints)]
