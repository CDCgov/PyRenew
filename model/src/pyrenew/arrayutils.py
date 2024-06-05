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
    fix_y: bool = False,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Pad the shorter array at the end to match the length of the longer array.

    Parameters
    ----------
    x : ArrayLike
        First array.
    y : ArrayLike
        Second array.
    fill_value : float, optional
        Value to use for padding, by default 0.0.
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
    if x_len > y_len:
        if fix_y:
            raise ValueError(
                "Cannot fix y when x is longer than y."
                + f" x_len: {x_len}, y_len: {y_len}."
            )

        y = jnp.pad(y, (0, x_len - y_len), constant_values=fill_value)

    elif y_len > x_len:
        x = jnp.pad(x, (0, y_len - x_len), constant_values=fill_value)

    return x, y


def pad_x_to_match_y(
    x: ArrayLike,
    y: ArrayLike,
    fill_value: float = 0.0,
) -> ArrayLike:
    """
    Pad the `x` array at the end to match the length of the `y` array.

    Parameters
    ----------
    x : ArrayLike
        First array.
    y : ArrayLike
        Second array.

    Returns
    -------
    Array
        Padded array.
    """
    return pad_to_match(x, y, fill_value=fill_value, fix_y=True)[0]


class PeriodicProcessSample(NamedTuple):
    """
    A container for holding the output from `process.PeriodicProcess.sample()`.

    Attributes
    ----------
    value : ArrayLike
        The sampled quantity.
    """

    value: ArrayLike | None = None

    def __repr__(self):
        return f"PeriodicProcessSample(value={self.value})"


class PeriodicBroadcaster:
    r"""
    Broadcast arrays periodically using either repeat or tile,
    considering period size and starting point.
    """

    def __init__(
        self,
        data_starts: int,
        period_size: int,
        broadcast_type: str,
    ) -> None:
        """
        Default constructor for PeriodicBroadcaster class.

        Parameters
        ----------
        data_starts : int
            Relative point at which data starts, must be between 0 and
            period_size - 1.
        period_size : int
            Size of the period.
        broadcast_type : str
            Type of broadcasting to use, either "repeat" or "tile".

        Notes
        -----
        See the sample method for more information on the broadcasting types.

        Returns
        -------
        None
        """

        self.validate(
            data_starts=data_starts,
            period_size=period_size,
            broadcast_type=broadcast_type,
        )

        self.period_size = period_size
        self.data_starts = data_starts
        self.broadcast_type = broadcast_type

        return None

    @staticmethod
    def validate(
        data_starts: int, period_size: int, broadcast_type: str
    ) -> None:
        """
        Validate the input parameters.

        Parameters
        ----------
        data_starts : int
            Relative point at which data starts, must be between 0 and
            period_size - 1.
        period_size : int
            Size of the period.
        broadcast_type : str
            Type of broadcasting to use, either "repeat" or "tile".

        Returns
        -------
        None
        """

        # Period size should be a positive integer
        assert isinstance(
            period_size, int
        ), f"period_size should be an integer. It is {type(period_size)}."

        assert (
            period_size > 0
        ), f"period_size should be a positive integer. It is {period_size}."

        # Data starts should be a positive integer
        assert isinstance(
            data_starts, int
        ), f"data_starts should be an integer. It is {type(data_starts)}."

        assert (
            0 <= data_starts
        ), f"data_starts should be a positive integer. It is {data_starts}."

        assert data_starts <= period_size - 1, (
            "data_starts should be less than or equal to period_size - 1."
            f"It is {data_starts}. It should be less than or equal "
            f"to {period_size - 1}."
        )

        # Broadcast type should be either "repeat" or "tile"
        assert broadcast_type in ["repeat", "tile"], (
            "broadcast_type should be either 'repeat' or 'tile'."
            f"It is {broadcast_type}."
        )

        return None

    def __call__(
        self,
        data: ArrayLike,
        n_timepoints: int,
    ) -> ArrayLike:
        """
        Broadcast the data to the given number of timepoints
        considering the period size and starting point.

        Parameters
        ----------
        data: ArrayLike
            Data to broadcast.
        n_timepoints : int
            Duration of the sequence.

        Notes
        -----
        The broadcasting is done by repeating or tiling the data. When
        self.broadcast_type = "repeat", the function will repeat each value of the data `self.period_size` times until it reaches `n_timepoints`. When self.broadcast_type = "tile", the function will tile the data until it reaches `n_timepoints`.

        Using the `data_starts` parameter, the function will start the broadcast from the `data_starts`-th element of the data. If the data is shorter than `n_timepoints`, the function will repeat or tile the data until it reaches `n_timepoints`.

        Returns
        -------
        ArrayLike
            Broadcasted array.
        """

        if self.broadcast_type == "repeat":
            return jnp.repeat(data, self.period_size)[
                self.data_starts : (self.data_starts + n_timepoints)
            ]
        else:
            return jnp.tile(
                data, int(jnp.ceil(n_timepoints / self.period_size))
            )[self.data_starts : (self.data_starts + n_timepoints)]
