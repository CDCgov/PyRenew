# numpydoc ignore=GL08
from abc import ABCMeta, abstractmethod

import jax.numpy as jnp
from jax.typing import ArrayLike

from pyrenew.metaclass import RandomVariable


class InfectionInitializationMethod(metaclass=ABCMeta):
    """Method for initializing infections in a renewal process."""

    def __init__(self, n_timepoints: int) -> None:
        """Default constructor for
        [`pyrenew.latent.infection_initialization_method.InfectionInitializationMethod`][].

        Parameters
        ----------
        n_timepoints
            the number of time points for which to
            generate initial infections

        Returns
        -------
        None
        """
        self.validate(n_timepoints)
        self.n_timepoints = n_timepoints

    @staticmethod
    def validate(n_timepoints: int) -> None:
        """
        Validate inputs to the
        [`pyrenew.latent.infection_initialization_method.InfectionInitializationMethod`][]
        constructor.

        Parameters
        ----------
        n_timepoints
            the number of time points to generate initial infections for

        Returns
        -------
        None
        """
        if not isinstance(n_timepoints, int):
            raise TypeError(
                f"n_timepoints must be an integer. Got {type(n_timepoints)}"
            )
        if n_timepoints <= 0:
            raise ValueError(f"n_timepoints must be positive. Got {n_timepoints}")

    @abstractmethod
    def initialize_infections(self, I_pre_init: ArrayLike) -> ArrayLike:
        """Generate the number of initialized infections at each time point.

        Parameters
        ----------
        I_pre_init
            An array representing some number of latent infections to be used with the specified `[`pyrenew.latent.infection_initialization_method.InfectionInitializationMethod`][]`.

        Returns
        -------
        ArrayLike
            An array of length ``n_timepoints`` with the number of initialized infections at each time point.
        """

    def __call__(self, I_pre_init: ArrayLike) -> ArrayLike:
        return self.initialize_infections(I_pre_init)


class InitializeInfectionsZeroPad(InfectionInitializationMethod):
    """
    Create an initial infection vector of specified length by
    padding a shorter vector with an appropriate number of
    zeros at the beginning of the time series.
    """

    def initialize_infections(self, I_pre_init: ArrayLike) -> ArrayLike:
        """Pad the initial infections with zeros at the beginning of the time series.

        Parameters
        ----------
        I_pre_init
            An array with initialized infections to be padded with zeros.

        Returns
        -------
        ArrayLike
            An array of length ``n_timepoints`` with the number of initialized infections at each time point.
        """
        I_pre_init = jnp.atleast_1d(I_pre_init)
        if self.n_timepoints < I_pre_init.size:
            raise ValueError(
                "I_pre_init must be no longer than n_timepoints. "
                f"Got I_pre_init of size {I_pre_init.size} and "
                f" n_timepoints of size {self.n_timepoints}."
            )
        return jnp.pad(I_pre_init, (self.n_timepoints - I_pre_init.size, 0))


class InitializeInfectionsFromVec(InfectionInitializationMethod):
    """Create initial infections from a vector of infections."""

    def initialize_infections(self, I_pre_init: ArrayLike) -> ArrayLike:
        """Create initial infections from a vector of infections.

        Parameters
        ----------
        I_pre_init
            An array with the same length as ``n_timepoints`` to be
            used as the initial infections.

        Returns
        -------
        ArrayLike
            An array of length ``n_timepoints`` with the number of
            initialized infections at each time point.
        """
        I_pre_init = jnp.array(I_pre_init)
        if I_pre_init.size != self.n_timepoints:
            raise ValueError(
                "I_pre_init must have the same size as n_timepoints. "
                f"Got I_pre_init of size {I_pre_init.size} "
                f"and n_timepoints of size {self.n_timepoints}."
            )
        return I_pre_init


class InitializeInfectionsExponentialGrowth(InfectionInitializationMethod):
    r"""Generate initial infections according to exponential growth.

    Notes
    -----
    The number of incident infections at time `t` is given by:

    ```math
    I(t) = I_p \exp \left( r (t - t_p) \right)
    ```

    Where $I_p$ is ``I_pre_init``, $r$ is ``rate``, and $t_p$ is ``t_pre_init``.
    This ensures that $I(t_p) = I_p$.
    We default to ``t_pre_init = n_timepoints - 1``, so that
    ``I_pre_init`` represents the number of incident infections immediately
    before the renewal process begins.
    """

    def __init__(
        self,
        n_timepoints: int,
        rate_rv: RandomVariable,
        t_pre_init: int | None = None,
    ) -> None:
        """Default constructor for the [`pyrenew.latent.infection_initialization_method.InitializeInfectionsExponentialGrowth`][] class.

        Parameters
        ----------
        n_timepoints
            the number of time points to generate initial infections for
        rate_rv
            A random variable representing the rate of exponential growth
        t_pre_init
             The time point whose number of infections is described by ``I_pre_init``. Defaults to ``n_timepoints - 1``.
        """
        super().__init__(n_timepoints)
        self.rate_rv = rate_rv
        if t_pre_init is None:
            t_pre_init = n_timepoints - 1
        self.t_pre_init = t_pre_init

    def initialize_infections(self, I_pre_init: ArrayLike) -> ArrayLike:
        """Generate initial infections according to exponential growth.

        Parameters
        ----------
        I_pre_init
            An array of size 1 representing the number of infections at time ``t_pre_init``.

        Returns
        -------
        ArrayLike
            An array of length ``n_timepoints`` with the number of initialized infections at each time point.
        """
        I_pre_init = jnp.array(I_pre_init)
        rate = jnp.array(self.rate_rv())
        initial_infections = I_pre_init * jnp.exp(
            rate * (jnp.arange(self.n_timepoints)[:, jnp.newaxis] - self.t_pre_init)
        )
        return jnp.squeeze(initial_infections)
