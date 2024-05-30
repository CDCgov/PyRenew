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

    @abstractmethod
    def seed_infections(self, I_pre_seed: ArrayLike):
        pass

    def __call__(self, I_pre_seed: ArrayLike):
        return self.seed_infections(I_pre_seed)


class SeedInfectionsZeroPad(InfectionSeedMethod):
    def seed_infections(self, I_pre_seed: ArrayLike):
        if self.n_timepoints < I_pre_seed.size:
            raise ValueError(
                f"I_pre_seed must be at least as long as n_timepoints. Got I_pre_seed of size {I_pre_seed.size} and and n_timepoints of size {self.n_timepoints}."
            )
        # alternative implementation:
        # return jnp.hstack([jnp.zeros(self.n_timepoints - I_pre_seed.size), I_pre_seed])
        return jnp.pad(I_pre_seed, (self.n_timepoints - I_pre_seed.size, 0))


class SeedInfectionsFromVec(InfectionSeedMethod):
    def seed_infections(self, I_pre_seed: ArrayLike):
        if I_pre_seed.size != self.n_timepoints:
            raise ValueError(
                f"I_pre_seed must have the same size as n_timepoints. Got I_pre_seed of size {I_pre_seed.size} and and n_timepoints of size {self.n_timepoints}."
            )
        return jnp.array(I_pre_seed)


class SeedInfectionsExponential(InfectionSeedMethod):
    def __init__(
        self,
        n_timepoints: int,
        rate: RandomVariable,
        t_I_pre_seed: int | None = None,
    ):
        super().__init__(n_timepoints)
        self.rate = rate
        if t_I_pre_seed is None:
            t_I_pre_seed = n_timepoints - 1
        self.t_I_pre_seed = t_I_pre_seed

    def seed_infections(self, I_pre_seed: ArrayLike):
        if I_pre_seed.size != 1:
            raise ValueError(
                f"I_pre_seed must be an array of size 1. Got size {I_pre_seed.size}."
            )
        (rate,) = self.rate.sample()
        return I_pre_seed * jnp.exp(
            rate * (jnp.arange(self.n_timepoints) - self.t_I_pre_seed)
        )
