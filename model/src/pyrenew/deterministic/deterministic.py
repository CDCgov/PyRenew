# -*- coding: utf-8 -*-

from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class DeterministicVariable(RandomVariable):
    """A deterministic (degenerate) random variable. Useful to pass fixed
    quantities."""

    def __init__(
        self,
        vars: ArrayLike,
        label: str = "a_random_variable",
    ) -> None:
        """Default constructor

        Parameters
        ----------
        vars : ArrayLike
            An array with the fixed quantity.
        label : str
            A label to assign to the process.

        Returns
        -------
        None
        """

        self.validate(vars)
        self.vars = vars
        self.label = label

        return None

    @staticmethod
    def validate(vars: ArrayLike) -> None:
        if not isinstance(vars, ArrayLike):
            raise Exception("vars is not an array-like object.")

        return None

    def sample(
        self,
        **kwargs,
    ) -> tuple:
        """Retrieve the value of the deterministic Rv

        Parameters
        ----------
        **kwargs : dict, optional
            Ignored.

        Returns
        -------
        tuple
            Containing the stored values during construction.
        """

        return (self.vars,)
