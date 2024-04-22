# -*- coding: utf-8 -*-

from pyrenew.metaclass import RandomVariable


class DeterministicVariable(RandomVariable):
    """A deterministic (degenerate) random variable. Useful to pass fixed
    quantities."""

    def __init__(
        self,
        vars: tuple,
        label: str = "a_random_variable",
    ) -> None:
        """Default constructor

        Parameters
        ----------
        vars : tuple
            A tuple with arraylike objects.
        label : str, optional
            A label to assign to the process. Defaults to "a_random_variable"

        Returns
        -------
        None
        """

        self.validate(vars)
        self.vars = vars
        self.label = label

        return None

    @staticmethod
    def validate(vars: tuple) -> None:
        """
        Validates inputted to DeterministicPMF

        Parameters
        ----------
        vars : tuple
            A tuple with arraylike objects.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the inputted vars object is not a tuple.
        """
        if not isinstance(vars, tuple):
            raise Exception("vars is not a tuple")

        return None

    def sample(
        self,
        **kwargs,
    ) -> tuple:
        """Retrieve the value of the deterministic Rv

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        tuple
            Containing the stored values during construction.
        """

        return self.vars
