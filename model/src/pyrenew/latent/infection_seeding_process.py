# -*- coding: utf-8 -*-
# numpydoc ignore=GL08
import numpyro as npro
from pyrenew.latent.infection_seeding_method import InfectionSeedMethod
from pyrenew.metaclass import RandomVariable


class InfectionSeedingProcess(RandomVariable):
    r"""Infection Seeding Process"""

    def __init__(
        self,
        I_pre_seed_rv: RandomVariable,
        infection_seed_method: InfectionSeedMethod,
    ) -> None:
        InfectionSeedingProcess.validate(I_pre_seed_rv, infection_seed_method)

        self.I_pre_seed_rv = I_pre_seed_rv
        self.infection_seed_method = infection_seed_method

        return None

    @staticmethod
    def validate(
        I_pre_seed_rv: RandomVariable,
        infection_seed_method: InfectionSeedMethod,
    ) -> None:
        if not isinstance(I_pre_seed_rv, RandomVariable):
            raise TypeError(
                "I_pre_seed_rv must be an instance of RandomVariable"
            )
        if not isinstance(infection_seed_method, InfectionSeedMethod):
            raise TypeError(
                "infection_seed_method must be an instance of InfectionSeedMethod"
            )

    def sample(self) -> tuple:
        (I_pre_seed,) = self.I_pre_seed_rv.sample()
        infection_seeding = self.infection_seed_method(I_pre_seed)
        npro.deterministic("I_seed", infection_seeding)

        return (infection_seeding,)
