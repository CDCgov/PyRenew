# numpydoc ignore=GL08
"""
Temporary execution context for sampled ascertainment values.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from jax.typing import ArrayLike

_AscertainmentValues = Mapping[str, Mapping[str, ArrayLike]]

_current_ascertainment_values: ContextVar[_AscertainmentValues | None] = ContextVar(
    "current_ascertainment_values",
    default=None,
)


def _validate_name(name: str, parameter: str) -> None:
    """
    Validate a context lookup or mapping key name.
    """
    if not isinstance(name, str) or len(name) == 0:
        raise ValueError(
            f"{parameter} must be a non-empty string. "
            f"Got {type(name).__name__}: {name!r}"
        )


def _validate_ascertainment_values(values: _AscertainmentValues) -> None:
    """
    Validate the nested mapping used for sampled ascertainment values.
    """
    if not isinstance(values, Mapping):
        raise TypeError(
            "ascertainment context values must be a mapping from "
            "ascertainment model names to signal-value mappings."
        )

    for ascertainment_name, signal_values in values.items():
        _validate_name(ascertainment_name, "ascertainment model name")
        if not isinstance(signal_values, Mapping):
            raise TypeError(
                "ascertainment context values must map each ascertainment "
                "model name to a signal-value mapping."
            )
        for signal_name in signal_values:
            _validate_name(signal_name, "signal name")


@contextmanager
def ascertainment_context(values: _AscertainmentValues) -> Any:
    """
    Temporarily make sampled ascertainment values available to accessors.

    Parameters
    ----------
    values
        Mapping from ascertainment model name to signal-specific values.

    Yields
    ------
    None
        The context in which signal accessors can retrieve sampled values.
    """
    _validate_ascertainment_values(values)
    token = _current_ascertainment_values.set(values)
    try:
        yield
    finally:
        _current_ascertainment_values.reset(token)


def get_ascertainment_value(
    ascertainment_name: str,
    signal_name: str,
) -> ArrayLike:
    """
    Retrieve a sampled signal-specific ascertainment value from context.

    Parameters
    ----------
    ascertainment_name
        Name of the ascertainment model.
    signal_name
        Name of the signal.

    Returns
    -------
    ArrayLike
        The sampled ascertainment value for the requested signal.

    Raises
    ------
    RuntimeError
        If no ascertainment context is active or the requested value is absent.
    """
    _validate_name(ascertainment_name, "ascertainment_name")
    _validate_name(signal_name, "signal_name")

    values = _current_ascertainment_values.get()
    if values is None:
        raise RuntimeError(
            f"Ascertainment signal {signal_name!r} from model "
            f"{ascertainment_name!r} was requested before ascertainment "
            "values were sampled."
        )

    try:
        return values[ascertainment_name][signal_name]
    except KeyError as exc:
        raise RuntimeError(
            f"Ascertainment signal {signal_name!r} from model "
            f"{ascertainment_name!r} is not available in the current context."
        ) from exc
