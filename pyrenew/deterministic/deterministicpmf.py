# numpydoc ignore=GL08

from __future__ import annotations

from jax.typing import ArrayLike

from pyrenew.deterministic.deterministic import DeterministicVariable
from pyrenew.distutil import validate_discrete_dist_vector
from pyrenew.metaclass import RandomVariable


class DeterministicPMF(RandomVariable):
    """
    A deterministic (degenerate) random variable that represents a PMF."""

    def __init__(
        self,
        name: str,
        value: ArrayLike,
        tol: float = 1e-5,
    ) -> None:
        """
        Default constructor

        Automatically checks that the elements in `value` can be indeed
        considered to be a PMF by calling
        pyrenew.distutil.validate_discrete_dist_vector on each one of its
        entries.

        Parameters
        ----------
        name
            A name to assign to the variable.
        value
            An ArrayLike object.
        tol
            Passed to pyrenew.distutil.validate_discrete_dist_vector. Defaults
            to 1e-5.

        Returns
        -------
        None
        """
        value = validate_discrete_dist_vector(
            discrete_dist=value,
            tol=tol,
        )

        self.basevar = DeterministicVariable(
            name=name,
            value=value,
        )

        return None

    @staticmethod
    def validate(value: ArrayLike) -> None:
        """
        Validates input to DeterministicPMF

        Parameters
        ----------
        value
            An ArrayLike object.

        Returns
        -------
        None
        """
        return None

    def sample(
        self,
        **kwargs,
    ) -> ArrayLike:
        """
        Retrieves the deterministic PMF

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed through to internal `sample()`
            calls, if any

        Returns
        -------
        ArrayLike
        """

        return self.basevar.sample(**kwargs)

    def size(self) -> int:
        """
        Returns the size of the PMF

        Returns
        -------
        int
            The size of the PMF
        """

        return self.basevar.value.size
