# -*- coding: utf-8 -*-

"""
Classes for the infection seeding methods
"""
from abc import ABCMeta, abstractmethod

import jax.numpy as jnp
from jax.typing import ArrayLike


class InfectionSeedMethod(metaclass=ABCMeta):
    # def __call__(self, I0: ArrayLike, n_timepoints: int) -> ArrayLike:
    #     return self.seed_infections(I0, n_timepoints)

    @abstractmethod
    def seed_infections(self, I0: ArrayLike, n_timepoints: int) -> ArrayLike:
        pass


class SeedInfectionsZeroPad(InfectionSeedMethod):
    def seed_infections(self, I0: ArrayLike, n_timepoints: int):
        # Don't know why this doesn't work here.
        # When I tested pad vs hstack outside of the class,
        # they both worked and pad was faster.
        # return jnp.pad(I0, (n_timepoints - I0.size, 0))
        return jnp.hstack([jnp.zeros(n_timepoints - I0.size), I0])


class SeedInfectionsRepeat(InfectionSeedMethod):
    def seed_infections(self, I0: ArrayLike, n_timepoints: int):
        return jnp.tile(I0, n_timepoints)


# Defined as below, I don't see how we can sample `rate`, which we probably want to do.
class SeedInfectionsExponential(InfectionSeedMethod):
    def __init__(self, rate: ArrayLike):
        self.rate = rate

    def seed_infections(self, I0: ArrayLike, n_timepoints: int):
        return I0 * jnp.exp(self.rate * jnp.arange(n_timepoints))
