# numpydoc ignore=GL08

from __future__ import annotations

from typing import Any

from jax.typing import ArrayLike

from pyrenew.deterministic.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable


class NullVariable(DeterministicVariable):
    """A null (degenerate) random variable. Sampling returns None."""

    def __init__(self) -> None:
        """Default constructor.

        Returns
        -------
        None
        """
        RandomVariable.__init__(self, name="null")
        self.validate()

        return None

    @staticmethod
    def validate() -> None:
        """
        Not used

        Returns
        -------
        None
        """
        return None

    def sample(
        self,
        **kwargs: Any,
    ) -> None:
        """Retrieve the value of the Null (None)

        Parameters
        ----------
        **kwargs
            Ignored.

        Returns
        -------
        None
        """

        return None


class NullObservation(NullVariable):
    """A null observation random variable. Sampling returns None."""

    def __init__(self) -> None:
        """Default constructor.

        Returns
        -------
        None
        """
        RandomVariable.__init__(self, name="null_observation")
        self.validate()

        return None

    @staticmethod
    def validate() -> None:
        """
        Not used

        Returns
        -------
        None
        """
        return None

    def sample(
        self,
        mu: ArrayLike,
        obs: ArrayLike | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Retrieve the value of the Null (None)

        Parameters
        ----------
        mu
            Unused parameter, represents mean of non-null distributions
        obs
            Observed data. Defaults to None.
        **kwargs
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        None
        """

        return None
