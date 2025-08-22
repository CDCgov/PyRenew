# numpydoc ignore=GL08
from __future__ import annotations

from jax.typing import ArrayLike

from pyrenew.latent.infection_initialization_method import (
    InfectionInitializationMethod,
)
from pyrenew.metaclass import RandomVariable, _assert_type


class InfectionInitializationProcess(RandomVariable):
    """Generate an initial infection history"""

    def __init__(
        self,
        name,
        I_pre_init_rv: RandomVariable,
        infection_init_method: InfectionInitializationMethod,
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

        Returns
        -------
        None
        """
        InfectionInitializationProcess.validate(I_pre_init_rv, infection_init_method)

        self.I_pre_init_rv = I_pre_init_rv
        self.infection_init_method = infection_init_method
        self.name = name

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
        _assert_type("I_pre_init_rv", I_pre_init_rv, RandomVariable)
        _assert_type(
            "infection_init_method",
            infection_init_method,
            InfectionInitializationMethod,
        )

    def sample(self) -> ArrayLike:
        """Sample the Infection Initialization Process.

        Returns
        -------
        ArrayLike
            the number of initialized infections at each time point.
        """

        I_pre_init = self.I_pre_init_rv()

        infection_initialization = self.infection_init_method(
            I_pre_init,
        )

        return infection_initialization
