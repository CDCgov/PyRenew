"""
Utility functions for processing arrays.
"""

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


def validate_arraylike(obj_to_validate: any, obj_name: str) -> None:
    """
    Validate that a passed argument is jax.typing.ArrayLike,
    raising an informative error if it is not.

    Parameters
    ----------
    obj_to_validate : any
        Object to validate.

    obj_name : str
        Name of the object to validate,
        for the error message if validation
        fails.

    Returns
    -------
    None
    """
    if not isinstance(obj_to_validate,
                      ArrayLike):
        raise ValueError(f"{obj_name} must be a JAX array "
                         "or behave like one, got "
                         f"{type(obj_to_validate)}."
                         "See documentation for jax.typing.ArrayLike"
                         "for more.")
