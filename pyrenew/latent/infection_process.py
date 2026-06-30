# numpydoc ignore=GL08

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

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


@runtime_checkable
class InfectionProcess(Protocol):
    """
    Protocol for infection processes.

    Infection processes convert a reproduction-number trajectory, initial
    infections, and generation interval into post-initialization infections.
    Implementations may be deterministic or may sample additional NumPyro
    random variables. Implementations should return the effective Rt used
    to generate post-initialization infections in ``InfectionProcessSample.rt``.
    """

    def sample(
        self,
        Rt: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        **kwargs: object,
    ) -> InfectionProcessSample:
        """
        Sample or compute infections from Rt, I0, and generation interval.
        """
        ...
