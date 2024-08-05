# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable, SampledValue


class FirstDifferencedProcess(RandomVariable):
    """
    Class for differenced stochastic process X(t),
    constructed by placing a fundamental stochastic
    process on the first differences (rates of change).
    """

    def __init__(
        self,
        name: str,
        fundamental_process: RandomVariable,
        init_value: ArrayLike,
        **kwargs,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name : str
            Name of the stochastic process
        fundamental_process : RandomVariable
            Stochastic process for the
            first differences
        init_value : ArrayLike
            Initial value of the process
        **kwargs :
            Additional keyword arguments passed to
            the parent class constructor.

        Returns
        -------
        None
        """
        self.name = name
        self.fundamental_process = fundamental_process
        self.init_value
        super().__init__(**kwargs)

    def sample(
        self,
        **kwargs,
    ) -> tuple:
        """
        Sample from the process

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments passed to self.fundamental_process.sample()

        Returns
        -------
        SampledValue
            Whose value entry is a single array representing the undifferenced
            timeseries
        """
        diffs = self.fundamental_process.sample()
        return SampledValue(
            value=jnp.cumsum(jnp.hstack[self.init_value, diffs.flatten()]),
            t_start=self.t_start,
            t_unit=self.t_unit,
        )

    @staticmethod
    def validate():
        """
        Validates input parameters, implementation pending.
        """
        return None
