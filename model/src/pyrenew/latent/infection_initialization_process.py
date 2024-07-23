# -*- coding: utf-8 -*-
# numpydoc ignore=GL08
import numpyro as npro
from pyrenew.latent.infection_initialization_method import (
    InfectionInitializationMethod,
)
from pyrenew.metaclass import RandomVariable


class InfectionInitializationProcess(RandomVariable):
    """Generate an initial infection history"""

    def __init__(
        self,
        name,
        I_pre_init_rv: RandomVariable,
        infection_init_method: InfectionInitializationMethod,
        t_unit: int,
        t_start: int | None = None,
    ) -> None:
        """Default class constructor for InfectionInitializationProcess

        Parameters
        ----------
        name : str
            A name to assign to the RandomVariable.
        I_pre_init_rv : RandomVariable
            A RandomVariable representing the number of infections that occur at some time before the renewal process begins. Each `infection_init_method` uses this random variable in different ways.
        infection_init_method : InfectionInitializationMethod
            An `InfectionInitializationMethod` that generates the initial infections for the renewal process.
        t_unit : int
            The unit of time for the time series passed to `RandomVariable.set_timeseries`.
        t_start : int, optional
            The relative starting time of the time series. If `None`, the relative starting time is set to `-infection_init_method.n_timepoints`.

        Notes
        -----
        The relative starting time of the time series (`t_start`) is set to `-infection_init_method.n_timepoints`.

        Returns
        -------
        None
        """
        InfectionInitializationProcess.validate(
            I_pre_init_rv, infection_init_method
        )

        self.I_pre_init_rv = I_pre_init_rv
        self.infection_init_method = infection_init_method
        self.name = name
        if t_start is None:
            t_start = -infection_init_method.n_timepoints

        self.set_timeseries(
            t_start=t_start,
            t_unit=t_unit,
        )

    @staticmethod
    def validate(
        I_pre_init_rv: RandomVariable,
        infection_init_method: InfectionInitializationMethod,
    ) -> None:
        """Validate the input arguments to the InfectionInitializationProcess class constructor

        Parameters
        ----------
        I_pre_init_rv : RandomVariable
            A random variable representing the number of infections that occur at some time before the renewal process begins.
        infection_init_method : InfectionInitializationMethod
            An method to generate the initial infections.

        Returns
        -------
        None
        """
        if not isinstance(I_pre_init_rv, RandomVariable):
            raise TypeError(
                "I_pre_init_rv must be an instance of RandomVariable"
                f"Got {type(I_pre_init_rv)}"
            )
        if not isinstance(
            infection_init_method, InfectionInitializationMethod
        ):
            raise TypeError(
                "infection_init_method must be an instance of InfectionInitializationMethod"
                f"Got {type(infection_init_method)}"
            )

    def sample(self) -> tuple:
        """Sample the Infection Initialization Process.

        Returns
        -------
        tuple
            a tuple where the only element is an array with the number of seeded infections at each time point.
        """
        (I_pre_init,) = self.I_pre_init_rv()
        infection_initialization = self.infection_init_method(I_pre_init)
        npro.deterministic(self.name, infection_initialization)

        return (infection_initialization,)
