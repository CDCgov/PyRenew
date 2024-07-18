# numpydoc ignore=GL08

from __future__ import annotations

from jax.typing import ArrayLike
from pyrenew.deterministic.deterministic import DeterministicVariable
from pyrenew.metaclass import TimeArray


class NullVariable(DeterministicVariable):
    """A null (degenerate) random variable. Sampling returns None."""

    def __init__(self) -> None:
        """Default constructor

        Returns
        -------
        None
        """

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
        **kwargs,
    ) -> tuple:
        """Retrieve the value of the Null (None)

        Parameters
        ----------.
        **kwargs : dict, optional
            Ignored.

        Returns
        -------
        tuple
            Containing a TimeArray with None.
        """

        return (TimeArray(None),)


class NullProcess(NullVariable):
    """A null random variable. Sampling returns None."""

    def __init__(self) -> None:
        """Default constructor

        Returns
        -------
        None
        """

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
        duration: int,
        **kwargs,
    ) -> tuple:
        """Retrieve the value of the Null (None)

        Parameters
        ----------
        duration : int
            Number of timepoints to sample (ignored).
        **kwargs : dict, optional
            Ignored.

        Returns
        -------
        tuple
            Containing a TimeArray with None.
        """

        return (TimeArray(None),)


class NullObservation(NullVariable):
    """A null observation random variable. Sampling returns None."""

    def __init__(self) -> None:
        """Default constructor

        Returns
        -------
        None
        """

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
        name: str | None = None,
        **kwargs,
    ) -> tuple:
        """
        Retrieve the value of the Null (None)

        Parameters
        ----------
        mu : ArrayLike
            Unused parameter, represents mean of non-null distributions
        obs : ArrayLike, optional
            Observed data. Defaults to None.
        name : str, optional
            Name of the random variable. Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        tuple
            Containing a TimeArray with None.
        """

        return (TimeArray(None),)
