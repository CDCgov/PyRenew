# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable
from pyrenew.process import ARProcess


class FirstDifferenceARProcess(RandomVariable):
    """
    Class for a stochastic process
    with an AR(1) process on the first
    differences (i.e. the rate of change).
    """

    def __init__(
        self,
        autoreg: ArrayLike,
        noise_sd: float,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        autoreg : ArrayLike
            Process parameters pyrenew.processesARprocess.
        noise_sd : float
            Error passed to pyrenew.processes.ARProcess.

        Returns
        -------
        None
        """
        self.rate_of_change_proc = ARProcess(0, jnp.array([autoreg]), noise_sd)

    def sample(
        self,
        duration: int,
        init_val: ArrayLike = None,
        init_rate_of_change: ArrayLike = None,
        name: str = "trend_rw",
        **kwargs,
    ) -> tuple:
        """
        Sample from the process

        Parameters
        ----------
        duration : int
            Passed to ARProcess.sample().s
        init_val : ArrayLike, optional
            Starting point of the AR process, by default None.
        init_rate_of_change : ArrayLike, optional
            Passed to ARProcess.sample, by default None.
        name : str, optional
            Passed to ARProcess.sample(), by default "trend_rw"
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        tuple
        """
        rates_of_change, *_ = self.rate_of_change_proc.sample(
            duration=duration,
            inits=jnp.atleast_1d(init_rate_of_change),
            name=name + "_rate_of_change",
        )
        return (init_val + jnp.cumsum(rates_of_change.flatten()),)

    @staticmethod
    def validate():
        """
        Validates inputted parameters, implementation pending.
        """
        return None
