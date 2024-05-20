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
            raise TypeError("n_timepoints must be an integer")
        if n_timepoints <= 0:
            raise ValueError("n_timepoints must be positive")

    @abstractmethod
    def seed_infections(self, I0: ArrayLike) -> ArrayLike:
        pass


class SeedInfectionsZeroPad(InfectionSeedMethod):
    def seed_infections(self, I0: ArrayLike):
        # alternative implementation:
        # return jnp.hstack([jnp.zeros(self.n_timepoints - I0.size), I0])
        return jnp.pad(I0, (self.n_timepoints - I0.size, 0))


class SeedInfectionsRepeat(InfectionSeedMethod):
    def seed_infections(self, I0: ArrayLike):
        return jnp.tile(I0, self.n_timepoints)


class SeedInfectionsExponential(InfectionSeedMethod):
    def __init__(self, n_timepoints: int, rate: RandomVariable):
        super().__init__(n_timepoints)
        self.rate = rate

    def seed_infections(self, I0: ArrayLike):
        # How do I ensure rate is recorded?
        rate = jnp.array(self.rate.sample())
        return I0 * jnp.exp(rate * jnp.arange(self.n_timepoints))


# TODO define __call__ so that you don't have to run the seed_infections functions
