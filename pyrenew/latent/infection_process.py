# numpydoc ignore=GL08

from __future__ import annotations

from typing import NamedTuple

from jax.typing import ArrayLike


class InfectionProcessSample(NamedTuple):
    """
    Output from an infection process.

    Attributes
    ----------
    post_initialization_infections
        The estimated latent infections after the initialization period.
    rt
        The effective reproduction number used by the infection process.
        For plain renewal processes this is the input Rt; processes with
        feedback or depletion may return an adjusted Rt.
    """

    post_initialization_infections: ArrayLike
    rt: ArrayLike
