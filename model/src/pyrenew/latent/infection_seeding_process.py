# -*- coding: utf-8 -*-
# numpydoc ignore=GL08
import numpyro as npro
from pyrenew.latent.infection_seeding_method import InfectionSeedMethod
from pyrenew.metaclass import RandomVariable


class InfectionSeedingProcess(RandomVariable):
    """Generate an initial infection history"""

    def __init__(
        self,
        I_pre_seed_rv: RandomVariable,
        infection_seed_method: InfectionSeedMethod,
    ) -> None:
        """Default class constructor for InfectionSeedingProcess

        Parameters
        ----------
        I_pre_seed_rv : RandomVariable
            A RandomVariable representing the number of infections that occur at some time before the renewal process begins. Each `infection_seed_method` uses this random variable in different ways.
        infection_seed_method : InfectionSeedMethod
            An `InfectionSeedMethod` that generates the seed infections for the renewal process.

        Returns
        -------
        None
        """
        InfectionSeedingProcess.validate(I_pre_seed_rv, infection_seed_method)

        self.I_pre_seed_rv = I_pre_seed_rv
        self.infection_seed_method = infection_seed_method


    @staticmethod
    def validate(
        I_pre_seed_rv: RandomVariable,
        infection_seed_method: InfectionSeedMethod,
    ) -> None:
        """Validate the input arguments to the InfectionSeedingProcess class constructor

        Parameters
        ----------
        I_pre_seed_rv : RandomVariable
            A random variable representing the number of infections that occur at some time before the renewal process begins.
        infection_seed_method : InfectionSeedMethod
            An method to generate the seed infections.

        Returns
        -------
        None
        """
        if not isinstance(I_pre_seed_rv, RandomVariable):
            raise TypeError(
                "I_pre_seed_rv must be an instance of RandomVariable"
            )
        if not isinstance(infection_seed_method, InfectionSeedMethod):
            raise TypeError(
                "infection_seed_method must be an instance of InfectionSeedMethod"
            )

    def sample(self) -> tuple:
        """Sample the infection seeding process.

        Returns
        -------
        tuple
            a tuple where the only element is an array with the number of seeded infections at each time point.
        """
        (I_pre_seed,) = self.I_pre_seed_rv.sample()
        infection_seeding = self.infection_seed_method(I_pre_seed)
        npro.deterministic("infection_seeding", infection_seeding)

        return (infection_seeding,)
