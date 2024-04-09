# -*- coding: utf-8 -*-

from jax.typing import ArrayLike
from pyrenew.deterministic.deterministic import DeterministicVariable
from pyrenew.distutil import validate_discrete_dist_vector
from pyrenew.metaclass import RandomVariable


class DeterministicPMF(RandomVariable):
    """A deterministic (degenerate) random variable that represents a PMF."""

    def __init__(
        self,
        vars: ArrayLike,
        label: str = "a_random_variable",
        tol: float = 1e-20,
    ) -> None:
        """Default constructor

        It automatically checks that the elements in `vars` can be indeed
        consireded to be a PMF by calling
        pyrenew.distutil.validate_discrete_dist_vector on each one of its
        entries.

        Parameters
        ----------
        vars : ArrayLike
            An array with the fixed quantity.
        label : str
            A label to assign to the process.
        tol : float
            Passed to pyrenew.distutil.validate_discrete_dist_vector

        Returns
        -------
        None
        """
        vars = validate_discrete_dist_vector(
            discrete_dist=vars,
            tol=tol,
        )

        self.basevar = DeterministicVariable(vars, label)

        return None

    @staticmethod
    def validate(vars: tuple) -> None:
        return None

    def sample(
        self,
        **kwargs,
    ) -> tuple:
        """Retrieves the deterministic PMF

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal `sample()`
            calls, if any

        Returns
        -------
        tuple
            Containing the stored values during construction.
        """

        return self.basevar.sample(**kwargs)
