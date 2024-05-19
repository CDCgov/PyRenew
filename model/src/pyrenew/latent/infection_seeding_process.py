# -*- coding: utf-8 -*-
# numpydoc ignore=GL08
import numpyro as npro
import numpyro.distributions as dist
from pyrenew.latent.infection_seeding_method import InfectionSeedMethod
from pyrenew.metaclass import RandomVariable


class InfectionSeedingProcess(RandomVariable):
    r"""Infection Seeding Process"""

    def __init__(
        self,
        I0_dist: dist.Distribution,
        infection_seed_method: InfectionSeedMethod,
    ) -> None:
        InfectionSeedingProcess.validate(I0_dist, infection_seed_method)

        self.I0_dist = I0_dist
        self.infection_seed_method = infection_seed_method

        return None

    @staticmethod
    def validate(
        I0_dist: dist.Distribution,
        infection_seed_method: InfectionSeedMethod,
    ) -> None:
        assert isinstance(I0_dist, dist.Distribution)
        assert isinstance(infection_seed_method, InfectionSeedMethod)

    def sample(self) -> tuple:
        I0 = npro.sample("I0", self.I0_dist)

        infection_seeding = self.infection_seed_method.seed_infections(I0)
        I0_det = npro.deterministic("I0", infection_seeding)

        return I0_det
