# -*- coding: utf-8 -*-

"""
pyrenew helper classes
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, NamedTuple, Self, get_type_hints

import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import Reparam

from pyrenew.mcmcutils import plot_posterior, spread_draws
from pyrenew.transformation import Transform


def _assert_type(arg_name: str, value, expected_type) -> None:
    """
    Matches TypeError arising during validation

    Parameters
    ----------
    arg_name : str
        Name of the argument
    value : object
        The object to be validated
    expected_type : type
        The expected object type

    Raises
    -------
    TypeError
        If `value` is not an instance of `expected_type`.

    Returns
    -------
    None
    """

    if not isinstance(value, expected_type):
        raise TypeError(
            f"{arg_name} must be an instance of {expected_type}. "
            f"Got {type(value)}"
        )


def _assert_sample_and_rtype(
    rp: "RandomVariable", skip_if_none: bool = True
) -> None:
    """
    Return type-checking for RandomVariable's sample function

    Objects passed as `RandomVariable` should (a) have a `sample()` method that
    (b) returns either a tuple or a named tuple.

    Parameters
    ----------
    rp : RandomVariable
        Random variable to check.
    skip_if_none : bool, optional
        When `True` it returns if `rp` is None. Defaults to True.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If rp is not a RandomVariable, does not have a sample function, or
        does not return a tuple. Also occurs if rettype does not initialized
        properly.
    """

    # Addressing the None case
    if (rp is None) and (not skip_if_none):
        Exception(
            "The passed object cannot be None. It should be RandomVariable"
        )
    elif skip_if_none and (rp is None):
        return None

    if not isinstance(rp, RandomVariable):
        raise Exception(f"{rp} is not an instance of RandomVariable.")

    # Otherwise, checking for the sample function (must have one)
    # with a defined rtype.
    try:
        sfun = rp.sample
    except Exception:
        raise Exception(
            f"The RandomVariable {rp} does not have a sample function."
        )  # noqa: E722

    # Getting the return annotation (if any)
    rettype = get_type_hints(sfun).get("return", None)

    if rettype is None:
        raise Exception(
            f"The RandomVariable {rp} does not have return type "
            + "annotation."
        )

    try:
        if not isinstance(rettype(), tuple):
            raise Exception(
                f"The RandomVariable {rp}'s return type annotation is not"
                + "a tuple"
            )
    except Exception:
        raise Exception(
            f"There was a problem when trying to initialize {rettype}."
            + "the rtype of the random variable should be a tuple or a namedtuple"
            + " with default values."
        )

    return None


class SampledValue(NamedTuple):
    """
    A container for a value sampled from a RandomVariable.

    Attributes
    ----------
    value : ArrayLike, optional
        The sampled value.
    t_start : int, optional
        The start time of the value.
    t_unit : int, optional
        The unit of time relative to the model's fundamental
        (smallest) time unit.
    """

    value: ArrayLike | None = None
    t_start: int | None = None
    t_unit: int | None = None

    def __repr__(self):
        return f"SampledValue(value={self.value}, t_start={self.t_start}, t_unit={self.t_unit})"


class RandomVariable(metaclass=ABCMeta):
    """
    Abstract base class for latent and observed random variables.

    Notes
    -----
    RandomVariables in pyrenew can be time-aware, meaning that they can
    have a t_start and t_unit attribute. These attributes
    are expected to be used internally mostly for tasks including padding,
    alignment of time series, and other time-aware operations.

    Both attributes give information about the output of the `sample()` method,
    in other words, the relative time units of the returning value.

    Attributes
    ----------
    t_start : int
        The start of the time series.
    t_unit : int
        The unit of the time series relative to the model's fundamental
        (smallest) time unit. e.g. if the fundamental unit is days,
        then 1 corresponds to units of days and 7 to units of weeks.
    """

    t_start: int = None
    t_unit: int = None

    def __init__(self, **kwargs):
        """
        Default constructor
        """
        pass

    def set_timeseries(
        self,
        t_start: int,
        t_unit: int,
    ) -> None:
        """
        Set the time series start and unit

        Parameters
        ----------
        t_start : int
            The start of the time series relative to the
            model time. It could be negative, indicating
            that the `sample()` method returns timepoints
            that occur prior to the model t = 0.
        t_unit : int
            The unit of the time series relative
            to the model's fundamental (smallest)
            time unit. e.g. if the fundamental unit
            is days, then 1 corresponds to units of
            days and 7 to units of weeks.

        Returns
        -------
        None
        """

        # Either both values are None or both are not None
        assert (t_unit is not None and t_start is not None) or (
            t_unit is None and t_start is None
        ), (
            "Both t_start and t_unit should be None or not None. "
            "Currently, t_start is {t_start} and t_unit is {t_unit}."
        )

        if t_unit is None and t_start is None:
            return None

        # Timeseries unit should be a positive integer
        assert isinstance(
            t_unit, int
        ), f"t_unit should be an integer. It is {type(t_unit)}."

        # Timeseries unit should be a positive integer
        assert (
            t_unit > 0
        ), f"t_unit should be a positive integer. It is {t_unit}."

        # Data starts should be a positive integer
        assert isinstance(
            t_start, int
        ), f"t_start should be an integer. It is {type(t_start)}."

        self.t_start = t_start
        self.t_unit = t_unit

        return None

    @abstractmethod
    def sample(
        self,
        **kwargs,
    ) -> tuple:
        """
        Sample method of the process

        The method design in the class should have at least kwargs.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal `sample()`
            calls, should there be any.

        Returns
        -------
        tuple
        """
        pass

    @staticmethod
    @abstractmethod
    def validate(**kwargs) -> None:
        """
        Validation of kwargs to be implemented in subclasses.
        """
        pass

    def __call__(self, **kwargs):
        """
        Alias for `sample()`.
        """
        return self.sample(**kwargs)


class DynamicDistributionalRV(RandomVariable):
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
        expand_shape: tuple = None,
    ) -> None:
        """
        Default constructor for DynamicDistributionalRV.

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
        expand_shape : tuple, optional
            If not None, expand the underlying distribution
            at sample() call by to the given expand_shape.
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
        if not (expand_shape is None or isinstance(expand_shape, tuple)):
            raise ValueError(
                "expand_shape must be a tuple or be None ",
                f"Got {type(expand_shape)}",
            )
        self.expand_shape = expand_shape

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
                "To instantiate a DynamicDistributionalRV, ",
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
            :fun:`numpyro.sample()`. Default `None`.
        **kwargs : dict, optional
            Keyword arguments passed to self.distribution_constructor

        Returns
        -------
        SampledValue
           Containing a sample from the distribution.
        """
        distribution = self.distribution_constructor(*args, **kwargs)
        if self.expand_shape is not None:
            distribution = distribution.expand(self.expand_shape)
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

    def expand(self, batch_shape) -> Self:
        """
        Expand the distribution to a different
        batch_shape, if possible. Returns a
        new DynamicDistributionalRV whose underlying
        distribution will be expanded by the given shape
        at sample() time.

        Parameters
        ----------
        batch_shape : tuple
            Batch shape for the expand. Will ultimately
            be passed to the expand() method of
            :class:`numpyro.distributions.Distribution`.

        Returns
        -------
        DynamicDistributionalRV
            Whose underlying distribution will be expanded to
            the desired batch shape at sampling time.
        """
        return DynamicDistributionalRV(
            name=self.name,
            distribution_constructor=self.distribution_constructor,
            reparam=self.reparam_dict.get(self.name, None),
            expand_shape=batch_shape,
        )


class StaticDistributionalRV(RandomVariable):
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
        Default constructor for DistributionalRV.

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
        Validation of the distribution to be implemented in subclasses.
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
            :fun:`numpyro.sample()`. Default `None`.
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

    def expand(self, batch_shape) -> Self:
        """
        Expand the distribution to a different
        batch_shape, if possible. Returns a
        new StaticDistributionalRV whose underlying
        distribution has been expanded by the given
        batch_shape.

        Parameters
        ----------
        batch_shape : tuple
            Batch shape for the expand. Passed to the expand()
            method of :class:`numpyro.distributions.Distribution`.

        Returns
        -------
        StaticDistributionalRV
            Whose underlying distribution has been expanded to
            the desired batch shape.
        """
        if not isinstance(batch_shape, tuple):
            raise ValueError(
                "batch_shape for expand()-ing "
                "a DistributionalRV must be a "
                f"tuple. Got {type(batch_shape)}"
            )
        return StaticDistributionalRV(
            name=self.name,
            distribution=self.distribution.expand(batch_shape),
            reparam=self.reparam_dict.get(self.name, None),
        )


def DistributionalRV(
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
        allows for dynamically-parameterized DistributionalRVs,
        e.g. a Normal distribution with an inferred location and
        scale.

    reparam : numpyro.infer.reparam.Reparam
        If not None, reparameterize sampling
        from the distribution according to the
        given numpyro reparameterizer

    Returns
    -------
    DynamicDistributionalRV | StaticDistributionalRV or
    raises a ValueError if a distribution cannot be constructed.
    """
    if isinstance(distribution, dist.Distribution):
        return StaticDistributionalRV(
            name=name, distribution=distribution, reparam=reparam
        )
    elif callable(distribution):
        return DynamicDistributionalRV(
            name=name, distribution_constructor=distribution, reparam=reparam
        )
    else:
        raise ValueError(
            "distribution argument to DistributionalRV "
            "must be either a numpyro.distributions.Distribution "
            "(for instantiating a static DistributionalRV) "
            "or a callable that returns a "
            "numpyro.distributions.Distribution (for "
            "a dynamic DistributionalRV"
        )


class Model(metaclass=ABCMeta):
    """Abstract base class for models"""

    # Since initialized in none, values not shared across instances
    kernel = None
    mcmc = None

    @abstractmethod
    def __init__(self, **kwargs) -> None:  # numpydoc ignore=GL08
        pass

    @staticmethod
    @abstractmethod
    def validate() -> None:  # numpydoc ignore=GL08
        pass

    @abstractmethod
    def sample(
        self,
        **kwargs,
    ) -> tuple:
        """
        Sample method of the model.

        The method design in the class should have at least kwargs.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal `sample()`
            calls, should there be any.

        Returns
        -------
        tuple
        """
        pass

    def model(self, **kwargs) -> tuple:
        """
        Alias for the sample method.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal `sample()`
            calls, should there be any.

        Returns
        -------
        tuple
        """
        return self.sample(**kwargs)

    def _init_model(
        self,
        num_warmup,
        num_samples,
        nuts_args: dict = None,
        mcmc_args: dict = None,
    ) -> None:
        """
        Creates the NUTS kernel and MCMC model

        Parameters
        ----------
        nuts_args : dict, optional
            Dictionary of arguments passed to NUTS. Defaults to None.
        mcmc_args : dict, optional
            Dictionary of arguments passed to the MCMC sampler. Defaults to
            None.

        Returns
        -------
        None
        """

        if nuts_args is None:
            nuts_args = dict()

        if mcmc_args is None:
            mcmc_args = dict()

        self.kernel = NUTS(
            model=self.model,
            **nuts_args,
        )

        self.mcmc = MCMC(
            self.kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            **mcmc_args,
        )

        return None

    def run(
        self,
        num_warmup,
        num_samples,
        rng_key: ArrayLike | None = None,
        nuts_args: dict = None,
        mcmc_args: dict = None,
        **kwargs,
    ) -> None:
        """
        Runs the model

        Parameters
        ----------
        nuts_args : dict, optional
            Dictionary of arguments passed to the
            :class:`numpyro.infer.NUTS` kernel.
            Defaults to None.
        mcmc_args : dict, optional
            Dictionary of arguments passed to the
            :class:`numpyro.infer.MCMC` constructor.
            Defaults to None.

        Returns
        -------
        None
        """

        self._init_model(
            num_warmup=num_warmup,
            num_samples=num_samples,
            nuts_args=nuts_args,
            mcmc_args=mcmc_args,
        )
        if rng_key is None:
            rand_int = np.random.randint(
                np.iinfo(np.int64).min, np.iinfo(np.int64).max
            )
            rng_key = jr.key(rand_int)

        self.mcmc.run(rng_key=rng_key, **kwargs)

        return None

    def print_summary(
        self,
        prob: float = 0.9,
        exclude_deterministic: bool = True,
    ) -> None:
        """
        A wrapper of :meth:`numpyro.infer.MCMC.print_summary`

        Parameters
        ----------
        prob : float, optional
            The width of the credible interval to show. Default 0.9
        exclude_deterministic : bool, optional
            Whether to print deterministic sites in the summary.
            Defaults to True.

        Returns
        -------
        None
        """
        return self.mcmc.print_summary(prob, exclude_deterministic)

    def spread_draws(self, variables_names: list) -> pl.DataFrame:
        """
        A wrapper of mcmcutils.spread_draws

        Parameters
        ----------
        variables_names : list
            A list of variable names to create a table of samples.

        Returns
        -------
        pl.DataFrame
        """

        return spread_draws(self.mcmc.get_samples(), variables_names)

    def plot_posterior(
        self,
        var: list,
        obs_signal: jax.typing.ArrayLike = None,
        xlab: str = None,
        ylab: str = "Signal",
        samples: int = 50,
        figsize: list = [4, 5],
        draws_col: str = "darkblue",
        obs_col: str = "black",
    ) -> plt.Figure:  # numpydoc ignore=RT01
        """A wrapper of pyrenew.mcmcutils.plot_posterior"""
        return plot_posterior(
            var=var,
            draws=self.spread_draws([(var, "time")]),
            xlab=xlab,
            ylab=ylab,
            samples=samples,
            obs_signal=obs_signal,
            figsize=figsize,
            draws_col=draws_col,
            obs_col=obs_col,
        )

    def posterior_predictive(
        self,
        rng_key: ArrayLike | None = None,
        numpyro_predictive_args: dict = {},
        **kwargs,
    ) -> dict:
        """
        A wrapper for :class:`numpyro.infer.Predictive` to generate
        posterior predictive samples.

        Parameters
        ----------
        rng_key : ArrayLike, optional
            Random key for the Predictive function call. Defaults to None.
        numpyro_predictive_args : dict, optional
            Dictionary of arguments to be passed to the
            :class:`numpyro.infer.Predictive` constructor.
        **kwargs
            Additional named arguments passed to the
            `__call__()` method of :class:`numpyro.infer.Predictive`

        Returns
        -------
        dict
        """
        if self.mcmc is None:
            raise ValueError(
                "No posterior samples available. Run model with model.run()."
            )

        if rng_key is None:
            rand_int = np.random.randint(
                np.iinfo(np.int64).min, np.iinfo(np.int64).max
            )
            rng_key = jr.key(rand_int)

        predictive = Predictive(
            model=self.model,
            posterior_samples=self.mcmc.get_samples(),
            **numpyro_predictive_args,
        )

        return predictive(rng_key, **kwargs)

    def prior_predictive(
        self,
        rng_key: ArrayLike | None = None,
        numpyro_predictive_args: dict = {},
        **kwargs,
    ) -> dict:
        """
        A wrapper for numpyro.infer.Predictive to generate prior predictive samples.

        Parameters
        ----------
        rng_key : ArrayLike, optional
            Random key for the Predictive function call. Defaults to None.
        numpyro_predictive_args : dict, optional
            Dictionary of arguments to be passed to the numpyro.infer.Predictive constructor.
        **kwargs
            Additional named arguments passed to the `__call__()` method of numpyro.infer.Predictive

        Returns
        -------
        dict
        """

        if rng_key is None:
            rand_int = np.random.randint(
                np.iinfo(np.int64).min, np.iinfo(np.int64).max
            )
            rng_key = jr.key(rand_int)

        predictive = Predictive(
            model=self.model,
            posterior_samples=None,
            **numpyro_predictive_args,
        )

        return predictive(rng_key, **kwargs)


class TransformedRandomVariable(RandomVariable):
    """
    Class to represent RandomVariables defined
    by taking the output of another RV's
    :meth:`RandomVariable.sample()` method
    and transforming it by a given transformation
    (typically a :class:`Transform`)
    """

    def __init__(
        self,
        name: str,
        base_rv: RandomVariable,
        transforms: Transform | tuple[Transform],
    ):
        """
        Default constructor

        Parameters
        ----------
        name : str
            A name for the random variable instance.
        base_rv : RandomVariable
            The underlying (untransformed) RandomVariable.
        transforms : Transform
            Transformation or tuple of transformations
            to apply to the output of
            `base_rv.sample()`; single values will be coerced to
            a length-one tuple. If a tuple, should be the same
            length as the tuple returned by `base_rv.sample()`.

        Returns
        -------
        None
        """
        self.name = name
        self.base_rv = base_rv

        if not isinstance(transforms, tuple):
            transforms = (transforms,)
        self.transforms = transforms
        self.validate()

    def sample(self, record=False, **kwargs) -> tuple:
        """
        Sample method. Call self.base_rv.sample()
        and then apply the transforms specified
        in self.transforms.

        Parameters
        ----------
        record : bool, optional
            Whether to record the value of the deterministic
            RandomVariable. Defaults to False.
        **kwargs :
            Keyword arguments passed to self.base_rv.sample()

        Returns
        -------
        tuple of the same length as the tuple returned by
        self.base_rv.sample()
        """

        untransformed_values = self.base_rv.sample(**kwargs)
        transformed_values = tuple(
            SampledValue(
                t(uv.value),
                t_start=self.t_start,
                t_unit=self.t_unit,
            )
            for t, uv in zip(self.transforms, untransformed_values)
        )

        if record:
            if len(untransformed_values) == 1:
                numpyro.deterministic(self.name, transformed_values[0].value)
            else:
                suffixes = (
                    untransformed_values._fields
                    if hasattr(untransformed_values, "_fields")
                    else range(len(transformed_values))
                )
                for suffix, tv in zip(suffixes, transformed_values):
                    numpyro.deterministic(f"{self.name}_{suffix}", tv.value)

        return transformed_values

    def sample_length(self):
        """
        Sample length for a transformed
        random variable must be equal to the
        length of self.transforms or
        validation will fail.

        Returns
        -------
        int
           Equal to the length self.transforms
        """
        return len(self.transforms)

    def validate(self):
        """
        Perform validation checks on a
        TransformedRandomVariable instance,
        confirming that all transformations
        are callable and that the number of
        transformations is equal to the sample
        length of the base random variable.

        Returns
        -------
        None
           on successful validation, or raise a ValueError
        """
        for t in self.transforms:
            if not callable(t):
                raise ValueError(
                    "All entries in self.transforms " "must be callable"
                )
        if hasattr(self.base_rv, "sample_length"):
            n_transforms = len(self.transforms)
            n_entries = self.base_rv.sample_length()
            if not n_transforms == n_entries:
                raise ValueError(
                    "There must be exactly as many transformations "
                    "specified as entries self.transforms as there are "
                    "entries in the tuple returned by "
                    "self.base_rv.sample()."
                    f"Got {n_transforms} transforms and {n_entries} "
                    "entries"
                )
