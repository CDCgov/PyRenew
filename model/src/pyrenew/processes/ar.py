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
from pyrenew.metaclasses import RandomProcess


class ARProcess(RandomProcess):
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

    def validate(self):
        return None
