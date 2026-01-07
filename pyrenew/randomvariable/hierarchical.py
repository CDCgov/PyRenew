# numpydoc ignore=GL08
"""
Hierarchical prior distributions for group-level random effects.

These classes provide random variables that sample from hierarchical
distributions with a `sample(n_groups=...)` interface, enabling
dynamic group sizes at sample time with proper numpyro plate contexts.
"""

import numpyro
import numpyro.distributions as dist

from pyrenew.metaclass import RandomVariable


class HierarchicalNormalPrior(RandomVariable):
    """
    Zero-centered Normal prior for group-level effects.

    Samples n_groups values from Normal(0, sd) within a numpyro plate context.

    Parameters
    ----------
    name : str
        Unique name for the sampled parameter in numpyro.
    sd_rv : RandomVariable
        RandomVariable returning the standard deviation.

    Notes
    -----
    This class is designed for hierarchical models where group effects
    are assumed to be drawn from a common distribution centered at zero.
    The number of groups is specified at sample time, allowing dynamic
    group sizes.

    Examples
    --------
    >>> from pyrenew.deterministic import DeterministicVariable
    >>> from pyrenew.randomvariable import HierarchicalNormalPrior
    >>> import numpyro
    >>>
    >>> sd_rv = DeterministicVariable("sd", 0.5)
    >>> prior = HierarchicalNormalPrior("site_effects", sd_rv)
    >>>
    >>> with numpyro.handlers.seed(rng_seed=42):
    ...     effects = prior.sample(n_groups=5)
    >>> effects.shape
    (5,)
    """

    def __init__(
        self,
        name: str,
        sd_rv: RandomVariable,
    ) -> None:
        """
        Default constructor for HierarchicalNormalPrior.

        Parameters
        ----------
        name : str
            Unique name for the sampled parameter in numpyro.
        sd_rv : RandomVariable
            RandomVariable returning the standard deviation.

        Returns
        -------
        None
        """
        if not isinstance(sd_rv, RandomVariable):
            raise TypeError(
                f"sd_rv must be a RandomVariable, got {type(sd_rv).__name__}. "
                "Use DeterministicVariable(name, value) to wrap a fixed value."
            )

        self.name = name
        self.sd_rv = sd_rv

    def validate(self):
        """Validate the random variable (no-op for this class)."""
        pass

    def sample(self, n_groups: int, **kwargs):
        """
        Sample group-level effects.

        Parameters
        ----------
        n_groups : int
            Number of groups.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        ArrayLike
            Array of shape (n_groups,) containing sampled effects.
        """
        sd = self.sd_rv()

        with numpyro.plate(f"n_{self.name}", n_groups):
            effects = numpyro.sample(
                self.name,
                dist.Normal(0.0, sd),
            )
        return effects


class TruncatedNormalGroupSdPrior(RandomVariable):
    """
    Truncated Normal prior for group-level standard deviations.

    Samples n_groups positive values from TruncatedNormal(loc, scale, low=sd_min)
    within a numpyro plate context. Uses NumPyro's native truncated distribution
    support for proper likelihood computation.

    Parameters
    ----------
    name : str
        Unique name for the sampled parameter in numpyro.
    loc_rv : RandomVariable
        RandomVariable returning the location (mean) of the underlying Normal.
    scale_rv : RandomVariable
        RandomVariable returning the scale (std) of the underlying Normal.
    sd_min : float, default=0.05
        Minimum SD value (left truncation point).

    Notes
    -----
    Truncated normal is a common choice for standard deviation priors because:

    - It naturally constrains values to be positive (via left truncation)
    - The location parameter represents the expected SD value
    - The scale parameter controls uncertainty about the SD
    - It integrates well with NumPyro's truncation support for proper likelihood

    Parameter guidance:

    - ``loc``: Set to your prior expectation for the SD (e.g., 0.3 for
      moderate sensor variability on log scale)
    - ``scale``: Controls uncertainty; smaller values give tighter priors
      around ``loc``
    - ``sd_min``: Prevents numerical issues when SDs approach zero; typical
      values are 0.01-0.1 depending on the scale of the data

    Examples
    --------
    >>> from pyrenew.deterministic import DeterministicVariable
    >>> from pyrenew.randomvariable import TruncatedNormalGroupSdPrior
    >>> import numpyro
    >>>
    >>> loc_rv = DeterministicVariable("sd_loc", 0.3)
    >>> scale_rv = DeterministicVariable("sd_scale", 0.15)
    >>> prior = TruncatedNormalGroupSdPrior("site_sd", loc_rv, scale_rv, sd_min=0.05)
    >>>
    >>> with numpyro.handlers.seed(rng_seed=42):
    ...     sds = prior.sample(n_groups=5)
    >>> all(sds >= 0.05)
    True
    """

    def __init__(
        self,
        name: str,
        loc_rv: RandomVariable,
        scale_rv: RandomVariable,
        sd_min: float = 0.05,
    ) -> None:
        """
        Default constructor for TruncatedNormalGroupSdPrior.

        Parameters
        ----------
        name : str
            Unique name for the sampled parameter in numpyro.
        loc_rv : RandomVariable
            RandomVariable returning the location (mean) of the underlying Normal.
        scale_rv : RandomVariable
            RandomVariable returning the scale (std) of the underlying Normal.
        sd_min : float, default=0.05
            Minimum SD value (left truncation point).

        Returns
        -------
        None
        """
        if not isinstance(loc_rv, RandomVariable):
            raise TypeError(
                f"loc_rv must be a RandomVariable, got {type(loc_rv).__name__}. "
                "Use DeterministicVariable(name, value) to wrap a fixed value."
            )
        if not isinstance(scale_rv, RandomVariable):
            raise TypeError(
                f"scale_rv must be a RandomVariable, got {type(scale_rv).__name__}. "
                "Use DeterministicVariable(name, value) to wrap a fixed value."
            )
        if sd_min < 0:
            raise ValueError(f"sd_min must be non-negative, got {sd_min}")

        self.name = name
        self.loc_rv = loc_rv
        self.scale_rv = scale_rv
        self.sd_min = sd_min

    def validate(self):
        """Validate the random variable (no-op for this class)."""
        pass

    def sample(self, n_groups: int, **kwargs):
        """
        Sample group-level standard deviations.

        Parameters
        ----------
        n_groups : int
            Number of groups.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        ArrayLike
            Array of shape (n_groups,) with values >= sd_min.
        """
        loc = self.loc_rv()
        scale = self.scale_rv()

        with numpyro.plate(f"n_{self.name}", n_groups):
            group_sd = numpyro.sample(
                self.name,
                dist.TruncatedNormal(loc=loc, scale=scale, low=self.sd_min),
            )
        return group_sd


class StudentTGroupModePrior(RandomVariable):
    """
    Zero-centered Student-t prior for group-level modes (robust alternative to Normal).

    Samples n_groups values from StudentT(df, 0, sd) within a numpyro plate context.
    This is useful when group effects may have heavier tails than a Normal distribution.

    Parameters
    ----------
    name : str
        Unique name for the sampled parameter in numpyro.
    sd_rv : RandomVariable
        RandomVariable returning the scale parameter.
    df_rv : RandomVariable
        RandomVariable returning the degrees of freedom.

    Notes
    -----
    The Student-t distribution approaches the Normal distribution as df -> infinity.
    Lower df values give heavier tails, making the prior more robust to outliers.
    Common choices include df=3 (heavy tails) or df=7 (moderate tails).

    Examples
    --------
    >>> from pyrenew.deterministic import DeterministicVariable
    >>> from pyrenew.randomvariable import StudentTGroupModePrior
    >>> import numpyro
    >>>
    >>> sd_rv = DeterministicVariable("scale", 0.5)
    >>> df_rv = DeterministicVariable("df", 4.0)
    >>> prior = StudentTGroupModePrior("site_modes", sd_rv, df_rv)
    >>>
    >>> with numpyro.handlers.seed(rng_seed=42):
    ...     modes = prior.sample(n_groups=5)
    >>> modes.shape
    (5,)
    """

    def __init__(
        self,
        name: str,
        sd_rv: RandomVariable,
        df_rv: RandomVariable,
    ) -> None:
        """
        Default constructor for StudentTGroupModePrior.

        Parameters
        ----------
        name : str
            Unique name for the sampled parameter in numpyro.
        sd_rv : RandomVariable
            RandomVariable returning the scale parameter.
        df_rv : RandomVariable
            RandomVariable returning the degrees of freedom.

        Returns
        -------
        None
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

        self.name = name
        self.sd_rv = sd_rv
        self.df_rv = df_rv

    def validate(self):
        """Validate the random variable (no-op for this class)."""
        pass

    def sample(self, n_groups: int, **kwargs):
        """
        Sample group-level modes.

        Parameters
        ----------
        n_groups : int
            Number of groups.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        ArrayLike
            Array of shape (n_groups,) containing sampled modes.
        """
        sd = self.sd_rv()
        df = self.df_rv()

        with numpyro.plate(f"n_{self.name}", n_groups):
            effects = numpyro.sample(
                self.name,
                dist.StudentT(df=df, loc=0.0, scale=sd),
            )
        return effects
