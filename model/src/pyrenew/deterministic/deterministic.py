# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import jax.numpy as jnp
import numpyro as npro
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable, SampledValue


class DeterministicVariable(RandomVariable):
    """
    A deterministic (degenerate) random variable. Useful to pass fixed
    quantities.
    """

    def __init__(
        self,
        vars: ArrayLike,
        name: str,
        t_start: int | None = None,
        t_unit: int | None = None,
    ) -> None:
        """Default constructor

        Parameters
        ----------
        vars : ArrayLike
            A tuple with arraylike objects.
        name : str, optional
            A name to assign to the process.
        t_start : int, optional
            The start time of the process.
        t_unit : int, optional
            The unit of time relative to the model's fundamental (smallest) time unit.

        Returns
        -------
        None
        """

        self.validate(vars)
        self.set_timeseries(t_start, t_unit)
        self.vars = jnp.atleast_1d(vars)
        self.name = name

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
            Containing the stored values during construction wrapped in a SampledValue.
        """
        if record:
            npro.deterministic(self.name, self.vars)
        return (
            SampledValue(
                value=self.vars,
                t_start=self.t_start,
                t_unit=self.t_unit,
            ),
        )
