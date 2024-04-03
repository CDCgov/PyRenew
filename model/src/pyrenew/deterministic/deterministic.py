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
    def validate(vars: tuple) -> None:
        if not isinstance(vars, tuple):
            raise Exception("vars is not a tuple")

        return None

    def sample(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        """Retrieve the value of the deterministic Rv

        Parameters
        ----------

        random_variables : dict
            Ignored. Default None.
        constants : dict
            Ignored. Default None.

        Returns
        -------
        tuple
            Containing the stored values during construction.
        """

        return self.vars
