# -*- coding: utf-8 -*-

"""
pyrenew helper classes
"""

from abc import ABCMeta, abstractmethod
from typing import get_type_hints

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import polars as pl
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import Reparam
from pyrenew.mcmcutils import plot_posterior, spread_draws
from pyrenew.transformation import Transform


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


class DistributionalRV(RandomVariable):
    """
    Wrapper class for random variables that sample
    from a single :class:`numpyro.distributions.Distribution`.
    """

    def __init__(
        self,
        dist: numpyro.distributions.Distribution,
        name: str,
        reparam: Reparam = None,
    ) -> None:
        """
        Default constructor for DistributionalRV.

        Parameters
        ----------
        dist : dist.Distribution
            Distribution of the random variable.
        name : str
            Name of the random variable.

        reparam : numpyro.infer.reparam.Reparam
            If not None, reparameterize sampling
            from the distribution according to the
            given numpyro reparameterizer

        Returns
        -------
        None
        """

        self.validate(dist)

        self.dist = dist
        self.name = name
        if reparam is not None:
            self.reparam_dict = {self.name: reparam}
        else:
            self.reparam_dict = {}

        return None

    @staticmethod
    def validate(dist: any) -> None:
        """
        Validation of the distribution to be implemented in subclasses.
        """
        if not isinstance(dist, numpyro.distributions.Distribution):
            raise ValueError(
                "dist should be an instance of "
                f"numpyro.distributions.Distribution, got {dist}"
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
        tuple
           Containing the sampled from the distribution.
        """
        with numpyro.handlers.reparam(config=self.reparam_dict):
            sample = numpyro.sample(
                name=self.name,
                fn=self.dist,
                obs=obs,
            )
        return (jnp.atleast_1d(sample),)


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

        if self.mcmc is None:
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
            :class:`numpyro.inference.Predictive` constructor.
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
            Dictionary of arguments to be passed to the numpyro.inference.Predictive constructor.
        **kwargs
            Additional named arguments passed to the `__call__()` method of numpyro.inference.Predictive

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
            A name for the random variable instance

        base_rv : RandomVariable
            The underlying (untransformed) RandomVariable

        transforms : Transform
            Transformation or tuple of transformations
            to apply to the output of
            `base_rv.sample()`; single values will be coerced to
            a length-one tuple. If a tuple, should be the same
            length as the tuple returned by `base_rv.sample()`

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

    def sample(self, **kwargs) -> tuple:
        """
        Sample method. Call self.base_rv.sample()
        and then apply the transforms specified
        in self.transforms.

        Parameters
        ----------
        **kwargs :
            Keyword arguments passed to self.base_rv.sample()

        Returns
        -------
        tuple of the same length as the tuple returned by
        self.base_rv.sample()
        """

        untransformed_values = self.base_rv.sample(**kwargs)

        return tuple(
            t(uv) for t, uv in zip(self.transforms, untransformed_values)
        )

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
