# numpydoc ignore=GL08

from typing import Callable, Self

import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.infer.reparam import Reparam

from pyrenew.metaclass import RandomVariable, SampledValue


class DynamicDistributionalVariable(RandomVariable):
    """
    Wrapper class for random variables that sample
    from a single :class:`numpyro.distributions.Distribution`
    that is parameterized / instantiated at `sample()` time
    (rather than at RandomVariable instantiation time).
    """

    def __init__(
        self,
        name: str,
        distribution_constructor: Callable,
        reparam: Reparam = None,
        expand_by_shape: tuple = None,
    ) -> None:
        """
        Default constructor for DynamicDistributionalVariable.

        Parameters
        ----------
        name : str
            Name of the random variable.
        distribution_constructor : Callable
            Callable that returns a concrete parametrized
            numpyro.Distributions.distribution instance.
        reparam : numpyro.infer.reparam.Reparam
            If not None, reparameterize sampling
            from the distribution according to the
            given numpyro reparameterizer
        expand_by_shape : tuple, optional
            If not None, call :meth:`expand_by()` on the
            underlying distribution once it is instianted
            with the given `expand_by_shape`.
            Default None.

        Returns
        -------
        None
        """

        self.name = name
        self.validate(distribution_constructor)
        self.distribution_constructor = distribution_constructor
        if reparam is not None:
            self.reparam_dict = {self.name: reparam}
        else:
            self.reparam_dict = {}
        if not (expand_by_shape is None or isinstance(expand_by_shape, tuple)):
            raise ValueError(
                "expand_by_shape must be a tuple or be None ",
                f"Got {type(expand_by_shape)}",
            )
        self.expand_by_shape = expand_by_shape

        return None

    @staticmethod
    def validate(distribution_constructor: any) -> None:
        """
        Confirm that the distribution_constructor is
        callable.

        Parameters
        ----------
        distribution_constructor : any
            Putative distribution_constructor to validate.

        Returns
        -------
        None or raises a ValueError
        """
        if not callable(distribution_constructor):
            raise ValueError(
                "To instantiate a DynamicDistributionalVariable, ",
                "one must provide a Callable that returns a "
                "numpyro.distributions.Distribution as the "
                "distribution_constructor argument. "
                f"Got {type(distribution_constructor)}, which "
                "does not appear to be callable",
            )
        return None

    def sample(
        self,
        *args,
        obs: ArrayLike = None,
        **kwargs,
    ) -> tuple:
        """
        Sample from the distributional rv.

        Parameters
        ----------
        *args :
            Positional arguments passed to self.distribution_constructor
        obs : ArrayLike, optional
            Observations passed as the `obs` argument to
            :meth:`numpyro.sample()`. Default `None`.
        **kwargs : dict, optional
            Keyword arguments passed to self.distribution_constructor

        Returns
        -------
        SampledValue
           Containing a sample from the distribution.
        """
        distribution = self.distribution_constructor(*args, **kwargs)
        if self.expand_by_shape is not None:
            distribution = distribution.expand_by(self.expand_by_shape)
        with numpyro.handlers.reparam(config=self.reparam_dict):
            sample = numpyro.sample(
                name=self.name,
                fn=distribution,
                obs=obs,
            )
        return (
            SampledValue(
                sample,
                t_start=self.t_start,
                t_unit=self.t_unit,
            ),
        )

    def expand_by(self, sample_shape) -> Self:
        """
        Expand the distribution by a given
        sample_shape, if possible. Returns a
        new DynamicDistributionalVariable whose underlying
        distribution will be expanded by the given shape
        at sample() time.

        Parameters
        ----------
        sample_shape : tuple
            Sample shape by which to expand the distribution.
            Passed to the expand_by() method of
            :class:`numpyro.distributions.Distribution`
            after the distribution is instantiated.

        Returns
        -------
        DynamicDistributionalVariable
            Whose underlying distribution will be expanded by
            the given sample shape at sampling time.
        """
        return DynamicDistributionalVariable(
            name=self.name,
            distribution_constructor=self.distribution_constructor,
            reparam=self.reparam_dict.get(self.name, None),
            expand_by_shape=sample_shape,
        )


class StaticDistributionalVariable(RandomVariable):
    """
    Wrapper class for random variables that sample
    from a single :class:`numpyro.distributions.Distribution`
    that is parameterized / instantiated at RandomVariable
    instantiation time (rather than at `sample()`-ing time).
    """

    def __init__(
        self,
        name: str,
        distribution: numpyro.distributions.Distribution,
        reparam: Reparam = None,
    ) -> None:
        """
        Default constructor for DistributionalVariable.

        Parameters
        ----------
        name : str
            Name of the random variable.
        distribution : numpyro.distributions.Distribution
            Distribution of the random variable.
        reparam : numpyro.infer.reparam.Reparam
            If not None, reparameterize sampling
            from the distribution according to the
            given numpyro reparameterizer

        Returns
        -------
        None
        """

        self.name = name
        self.validate(distribution)
        self.distribution = distribution
        if reparam is not None:
            self.reparam_dict = {self.name: reparam}
        else:
            self.reparam_dict = {}

        return None

    @staticmethod
    def validate(distribution: any) -> None:
        """
        Validation of the distribution.
        """
        if not isinstance(distribution, numpyro.distributions.Distribution):
            raise ValueError(
                "distribution should be an instance of "
                "numpyro.distributions.Distribution, got "
                "{type(distribution)}"
            )

        return None

    def sample(
        self,
        obs: ArrayLike | None = None,
        **kwargs,
    ) -> tuple:
        """
        Sample from the distribution.

        Parameters
        ----------
        obs : ArrayLike, optional
            Observations passed as the `obs` argument to
            :meth:`numpyro.sample()`. Default `None`.
        **kwargs : dict, optional
            Additional keyword arguments passed through
            to internal sample calls, should there be any.

        Returns
        -------
        SampledValue
           Containing a sample from the distribution.
        """
        with numpyro.handlers.reparam(config=self.reparam_dict):
            sample = numpyro.sample(
                name=self.name,
                fn=self.distribution,
                obs=obs,
            )
        return (
            SampledValue(
                sample,
                t_start=self.t_start,
                t_unit=self.t_unit,
            ),
        )

    def expand_by(self, sample_shape) -> Self:
        """
        Expand the distribution by the given sample_shape,
        if possible. Returns a new StaticDistributionalVariable
        whose underlying distribution has been expanded by
        the given sample_shape via
        :meth:`~numpyro.distributions.Distribution.expand_by()`

        Parameters
        ----------
        sample_shape : tuple
            Sample shape for the expansion. Passed to the
            :meth:`expand_by()` method of
            :class:`numpyro.distributions.Distribution`.

        Returns
        -------
        StaticDistributionalVariable
            Whose underlying distribution has been expanded by
            the given sample shape.
        """
        if not isinstance(sample_shape, tuple):
            raise ValueError(
                "sample_shape for expand()-ing "
                "a DistributionalVariable must be a "
                f"tuple. Got {type(sample_shape)}"
            )
        return StaticDistributionalVariable(
            name=self.name,
            distribution=self.distribution.expand_by(sample_shape),
            reparam=self.reparam_dict.get(self.name, None),
        )


def DistributionalVariable(
    name: str,
    distribution: numpyro.distributions.Distribution | Callable,
    reparam: Reparam = None,
) -> RandomVariable:
    """
    Factory function to generate Distributional RandomVariables,
    either static or dynamic.

    Parameters
    ----------
    name : str
        Name of the random variable.

    distribution: numpyro.distributions.Distribution | Callable
        Either numpyro.distributions.Distribution instance
        given the static distribution of the random variable or
        a callable that returns a parameterized
        numpyro.distributions.Distribution when called, which
        allows for dynamically-parameterized DistributionalVariables,
        e.g. a Normal distribution with an inferred location and
        scale.

    reparam : numpyro.infer.reparam.Reparam
        If not None, reparameterize sampling
        from the distribution according to the
        given numpyro reparameterizer

    Returns
    -------
    DynamicDistributionalVariable | StaticDistributionalVariable or
    raises a ValueError if a distribution cannot be constructed.
    """
    if isinstance(distribution, dist.Distribution):
        return StaticDistributionalVariable(
            name=name, distribution=distribution, reparam=reparam
        )
    elif callable(distribution):
        return DynamicDistributionalVariable(
            name=name, distribution_constructor=distribution, reparam=reparam
        )
    else:
        raise ValueError(
            "distribution argument to DistributionalVariable "
            "must be either a numpyro.distributions.Distribution "
            "(for instantiating a static DistributionalVariable) "
            "or a callable that returns a "
            "numpyro.distributions.Distribution (for "
            "a dynamic DistributionalVariable)."
        )
