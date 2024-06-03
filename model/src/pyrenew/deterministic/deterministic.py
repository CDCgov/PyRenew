# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import numpyro as npro
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class DeterministicVariable(RandomVariable):
    """
    A deterministic (degenerate) random variable. Useful to pass fixed
    quantities.
    """

    def __init__(
        self,
        vars: ArrayLike,
        label: str = "a_random_variable",
    ) -> None:
        """Default constructor

        Parameters
        ----------
        vars : ArrayLike
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
    def validate(vars: ArrayLike) -> None:
        """
        Validates inputted to DeterministicPMF

        Parameters
        ----------
        vars : ArrayLike
            An ArrayLike object.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the inputted vars object is not a ArrayLike.
        """
        if not isinstance(vars, ArrayLike):
            raise Exception("vars is not a ArrayLike")

        return None

    def sample(
        self,
        record=True,
        **kwargs,
    ) -> tuple:
        """
        Retrieve the value of the deterministic Rv

        Parameters
        ----------
        record : bool, optional
            Whether to record the value of the deterministic RandomVariable. Defaults to True.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        tuple
            Containing the stored values during construction.
        """
        if record:
            npro.deterministic(self.label, self.vars)
        return (self.vars,)
