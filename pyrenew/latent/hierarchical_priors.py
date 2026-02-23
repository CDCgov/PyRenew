"""
Hierarchical prior distributions for group-level random effects.

All classes in this module implement the group-level RV interface:
``sample(n_groups, **kwargs) -> ArrayLike`` with shape ``(n_groups,)``.
"""


import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from pyrenew.metaclass import RandomVariable


class HierarchicalNormalPrior(RandomVariable):
    """
    Zero-centered Normal prior for group-level effects.

    Samples n_groups values from Normal(0, sd).

    Parameters
    ----------
    name
        Unique name for the sampled parameter in numpyro.
    sd_rv
        RandomVariable returning the standard deviation.
    """

    def __init__(
        self,
        name: str,
        sd_rv: RandomVariable,
    ) -> None:
        """
        Initialize hierarchical normal prior.

        Parameters
        ----------
        name
            Unique name for the sampled parameter.
        sd_rv
            RandomVariable returning the standard deviation.
        """
        if not isinstance(sd_rv, RandomVariable):
            raise TypeError(
                f"sd_rv must be a RandomVariable, got {type(sd_rv).__name__}. "
                "Use DeterministicVariable(name, value) to wrap a fixed value."
            )

        super().__init__(name=name)
        self.sd_rv = sd_rv

    def validate(self) -> None:
        """Validate the random variable (no-op for this class)."""
        pass

    def sample(self, n_groups: int, **kwargs: object) -> ArrayLike:
        """
        Sample group-level effects.

        Parameters
        ----------
        n_groups
            Number of groups.

        Returns
        -------
        ArrayLike
            Array of shape (n_groups,).
        """
        sd = self.sd_rv()

        with numpyro.plate(f"n_{self.name}", n_groups):
            effects = numpyro.sample(
                self.name,
                dist.Normal(0.0, sd),
            )
        return effects


class GammaGroupSdPrior(RandomVariable):
    """
    Gamma prior for group-level standard deviations, bounded away from zero.

    Samples n_groups positive values from Gamma(concentration, rate) + sd_min.

    Parameters
    ----------
    name
        Unique name for the sampled parameter in numpyro.
    sd_mean_rv
        RandomVariable returning the mean of the Gamma distribution.
    sd_concentration_rv
        RandomVariable returning the concentration (shape) parameter of Gamma.
    sd_min
        Minimum SD value (lower bound).
    """

    def __init__(
        self,
        name: str,
        sd_mean_rv: RandomVariable,
        sd_concentration_rv: RandomVariable,
        sd_min: float = 0.05,
    ) -> None:
        """
        Initialize gamma group SD prior.

        Parameters
        ----------
        name
            Unique name for the sampled parameter.
        sd_mean_rv
            RandomVariable returning the mean of the Gamma distribution.
        sd_concentration_rv
            RandomVariable returning the concentration parameter.
        sd_min
            Minimum SD value (lower bound).
        """
        if not isinstance(sd_mean_rv, RandomVariable):
            raise TypeError(
                f"sd_mean_rv must be a RandomVariable, got {type(sd_mean_rv).__name__}. "
                "Use DeterministicVariable(name, value) to wrap a fixed value."
            )
        if not isinstance(sd_concentration_rv, RandomVariable):
            raise TypeError(
                f"sd_concentration_rv must be a RandomVariable, got {type(sd_concentration_rv).__name__}. "
                "Use DeterministicVariable(name, value) to wrap a fixed value."
            )
        if sd_min < 0:
            raise ValueError(f"sd_min must be non-negative, got {sd_min}")

        super().__init__(name=name)
        self.sd_mean_rv = sd_mean_rv
        self.sd_concentration_rv = sd_concentration_rv
        self.sd_min = sd_min

    def validate(self) -> None:
        """Validate the random variable (no-op for this class)."""
        pass

    def sample(self, n_groups: int, **kwargs: object) -> ArrayLike:
        """
        Sample group-level standard deviations.

        Parameters
        ----------
        n_groups
            Number of groups.

        Returns
        -------
        ArrayLike
            Array of shape (n_groups,) with values >= sd_min.
        """
        sd_mean = self.sd_mean_rv()
        concentration = self.sd_concentration_rv()
        rate = concentration / sd_mean

        with numpyro.plate(f"n_{self.name}", n_groups):
            raw_sd = numpyro.sample(
                f"{self.name}_raw",
                dist.Gamma(concentration, rate),
            )

            group_sd = numpyro.deterministic(
                self.name,
                jnp.maximum(raw_sd, self.sd_min),
            )
        return group_sd


class StudentTGroupModePrior(RandomVariable):
    """
    Zero-centered Student-t prior for group-level modes (robust alternative to Normal).

    Samples n_groups values from StudentT(df, 0, sd).

    Parameters
    ----------
    name
        Unique name for the sampled parameter in numpyro.
    sd_rv
        RandomVariable returning the scale parameter.
    df_rv
        RandomVariable returning the degrees of freedom.
    """

    def __init__(
        self,
        name: str,
        sd_rv: RandomVariable,
        df_rv: RandomVariable,
    ) -> None:
        """
        Initialize Student-t group mode prior.

        Parameters
        ----------
        name
            Unique name for the sampled parameter.
        sd_rv
            RandomVariable returning the scale parameter.
        df_rv
            RandomVariable returning the degrees of freedom.
        """
        if not isinstance(sd_rv, RandomVariable):
            raise TypeError(
                f"sd_rv must be a RandomVariable, got {type(sd_rv).__name__}. "
                "Use DeterministicVariable(name, value) to wrap a fixed value."
            )
        if not isinstance(df_rv, RandomVariable):
            raise TypeError(
                f"df_rv must be a RandomVariable, got {type(df_rv).__name__}. "
                "Use DeterministicVariable(name, value) to wrap a fixed value."
            )

        super().__init__(name=name)
        self.sd_rv = sd_rv
        self.df_rv = df_rv

    def validate(self) -> None:
        """Validate the random variable (no-op for this class)."""
        pass

    def sample(self, n_groups: int, **kwargs: object) -> ArrayLike:
        """
        Sample group-level modes.

        Parameters
        ----------
        n_groups
            Number of groups.

        Returns
        -------
        ArrayLike
            Array of shape (n_groups,).
        """
        sd = self.sd_rv()
        df = self.df_rv()

        with numpyro.plate(f"n_{self.name}", n_groups):
            effects = numpyro.sample(
                self.name,
                dist.StudentT(df=df, loc=0.0, scale=sd),
            )
        return effects
