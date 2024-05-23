from abc import ABCMeta, abstractmethod

import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

# -*- coding: utf-8 -*-
# numpydoc ignore=GL08
"""
Classes for the infection seeding methods
"""


class InfectionSeedMethod(metaclass=ABCMeta):
    def __init__(self, n_timepoints: int):
        self.validate(n_timepoints)
        self.n_timepoints = n_timepoints

    @staticmethod
    def validate(n_timepoints: int) -> None:
        if not isinstance(n_timepoints, int):
            raise TypeError(
                f"n_timepoints must be an integer. Got {type(n_timepoints)}"
            )
        if n_timepoints <= 0:
            raise ValueError(
                f"n_timepoints must be positive. Got {n_timepoints}"
            )

    abstractmethod

    def seed_infections(self, *args, **kwargs):
        pass

        pass

    def __call__(self, *args, **kwargs):
        return self.seed_infections(*args, **kwargs)


class SeedInfectionsZeroPad(InfectionSeedMethod):
    def seed_infections(self, I_last: ArrayLike):
        if self.n_timepoints < I_last.size:
            raise ValueError(
                f"I_last must be at least as long as n_timepoints. Got I_last of size {I_last.size} and and n_timepoints of size {self.n_timepoints}."
            )
        # alternative implementation:
        # return jnp.hstack([jnp.zeros(self.n_timepoints - I0.size), I0])
        return jnp.pad(I_last, (self.n_timepoints - I_last.size, 0))


class SeedInfectionsFromVec(InfectionSeedMethod):
    def seed_infections(self, I_seed: ArrayLike):
        if I_seed.size != self.n_timepoints:
            raise ValueError(
                f"I_seed must have the same size as n_timepoints. Got I_seed of size {I_seed.size} and and n_timepoints of size {self.n_timepoints}."
            )
        return jnp.array(I_seed)


class SeedInfectionsExponential(InfectionSeedMethod):
    def __init__(self, n_timepoints: int, rate: RandomVariable):
        super().__init__(n_timepoints)
        self.rate = rate

    def seed_infections(self, I0: ArrayLike):
        if I0.size != 1:
            raise ValueError(
                f"I0 must be an array of size 1. Got size {I0.size}."
            )
        rate = jnp.array(self.rate.sample())
        return I0 * jnp.exp(rate * jnp.arange(self.n_timepoints))
