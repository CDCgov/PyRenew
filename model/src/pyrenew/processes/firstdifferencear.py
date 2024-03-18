#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process

Pyrenew classes for common
stochastic processes
"""
import jax.numpy as jnp
from numpy.typing import ArrayLike
from pyrenew.metaclasses import RandomProcess
from pyrenew.processes import ARProcess


class FirstDifferenceARProcess(RandomProcess):
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
        """Default constructor

        :param autoreg: Process parameters pyrenew.processesARprocess.
        :type autoreg: ArrayLike
        :param noise_sd: Error passed to pyrenew.processes.ARProcess
        :type noise_sd: float
        """
        self.rate_of_change_proc = ARProcess(0, jnp.array([autoreg]), noise_sd)

    def sample(
        self,
        duration,
        init_val: ArrayLike = None,
        init_rate_of_change: ArrayLike = None,
        name: str = "trend_rw",
    ) -> tuple:
        rocs, *_ = self.rate_of_change_proc.sample(
            duration, inits=init_rate_of_change, name=name + "_rate_of_change"
        )
        return (init_val + jnp.cumsum(rocs.flatten()),)

    @staticmethod
    def validate():
        return None
