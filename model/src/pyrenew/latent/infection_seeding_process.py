# -*- coding: utf-8 -*-
# numpydoc ignore=GL08
import numpyro as npro
from pyrenew.latent.infection_seeding_method import (
    InfectionInitializationMethod,
)
from pyrenew.metaclass import RandomVariable


class InfectionInitializationProcess(RandomVariable):
    """Generate an initial infection history"""

    def __init__(
        self,
        name,
        I_pre_seed_rv: RandomVariable,
        infection_seed_method: InfectionInitializationMethod,
        t_unit: int,
        t_start: int | None = None,
    ) -> None:
        """Default class constructor for InfectionInitializationProcess

        Parameters
        ----------
        name : str
            A name to assign to the RandomVariable.
        I_pre_seed_rv : RandomVariable
            A RandomVariable representing the number of infections that occur at some time before the renewal process begins. Each `infection_seed_method` uses this random variable in different ways.
        infection_seed_method : InfectionInitializationMethod
            An `InfectionInitializationMethod` that generates the seed infections for the renewal process.
        t_unit : int
            The unit of time for the time series passed to `RandomVariable.set_timeseries`.
        t_start : int, optional
            The relative starting time of the time series. If `None`, the relative starting time is set to `-infection_seed_method.n_timepoints`.

        Notes
        -----
        The relative starting time of the time series (`t_start`) is set to `-infection_seed_method.n_timepoints`.

        Returns
        -------
        None
        """
        InfectionInitializationProcess.validate(
            I_pre_seed_rv, infection_seed_method
        )

        self.I_pre_seed_rv = I_pre_seed_rv
        self.infection_seed_method = infection_seed_method
        self.name = name
        if t_start is None:
            t_start = -infection_seed_method.n_timepoints

        self.set_timeseries(
            t_start=t_start,
            t_unit=t_unit,
        )

    @staticmethod
    def validate(
        I_pre_seed_rv: RandomVariable,
        infection_seed_method: InfectionInitializationMethod,
    ) -> None:
        """Validate the input arguments to the InfectionInitializationProcess class constructor

        Parameters
        ----------
        I_pre_seed_rv : RandomVariable
            A random variable representing the number of infections that occur at some time before the renewal process begins.
        infection_seed_method : InfectionInitializationMethod
            An method to generate the seed infections.

        Returns
        -------
        None
        """
        if not isinstance(I_pre_seed_rv, RandomVariable):
            raise TypeError(
                "I_pre_seed_rv must be an instance of RandomVariable"
                f"Got {type(I_pre_seed_rv)}"
            )
        if not isinstance(
            infection_seed_method, InfectionInitializationMethod
        ):
            raise TypeError(
                "infection_seed_method must be an instance of InfectionInitializationMethod"
                f"Got {type(infection_seed_method)}"
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
        npro.deterministic(self.name, infection_seeding)

        return (infection_seeding,)
