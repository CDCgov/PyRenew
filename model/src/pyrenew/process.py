#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process

Pyrenew classes for common
stochastic processes
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import lax
from numpy.typing import ArrayLike


class ARProcess:
    """
    Object to represent
    an AR(p) process in
    Numpyro
    """

    def __init__(self, mean: float, autoreg: ArrayLike, noise_sd: float):
        self.mean = mean
        self.autoreg = autoreg
        self.noise_sd = noise_sd

    def sample(self, duration, inits=None, name="arprocess"):
        order = self.autoreg.shape[0]
        if inits is None:
            inits = numpyro.sample(
                name + "_sampled_inits",
                dist.Normal(0, self.noise_sd).expand((order,)),
            )

        def _ar_scanner(carry, next):
            new_term = (jnp.dot(self.autoreg, carry) + next).flatten()
            new_carry = jnp.hstack([new_term, carry[: (order - 1)]])
            return new_carry, new_term

        noise = numpyro.sample(
            name + "_noise", dist.Normal(0, self.noise_sd).expand((duration,))
        )

        last, ts = lax.scan(_ar_scanner, inits - self.mean, noise)
        return self.mean + ts.flatten()


class FirstDifferenceARProcess:
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


class SimpleRandomWalk:
    """
    Class for a Markovian
    random walk with an a
    abitrary step distribution
    """

    def __init__(self, error_distribution: dist.Distribution):
        self.error_distribution = error_distribution

    def sample(self, duration, name="randomwalk", init=None):
        if init is None:
            init = numpyro.sample(name + "_init", self.error_distribution)
        diffs = numpyro.sample(
            name + "_diffs", self.error_distribution.expand((duration,))
        )

        return init + jnp.cumsum(jnp.pad(diffs, [1, 0], constant_values=0))
