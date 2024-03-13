#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process

Pyrenew classes for common
stochastic processes
"""
import jax.numpy as jnp
from pyrenew.metaclasses import RandomProcess
from pyrenew.processes import ARProcess


class FirstDifferenceARProcess(RandomProcess):
    """
    Class for a stochastic process
    with an AR(1) process on the first
    differences (i.e. the rate of change).
    """

    def __init__(self, autoreg, noise_sd):
        self.rate_of_change_proc = ARProcess(0, jnp.array([autoreg]), noise_sd)

    def sample(
        self,
        duration,
        init_val=None,
        init_rate_of_change=None,
        name="trend_rw",
    ):
        rocs = self.rate_of_change_proc.sample(
            duration, inits=init_rate_of_change, name=name + "_rate_of_change"
        )
        return init_val + jnp.cumsum(rocs.flatten())

    def validate(self):
        return None
