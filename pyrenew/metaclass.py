"""
pyrenew helper classes
"""

from abc import ABCMeta, abstractmethod

import jax.random as jr
import numpy as np
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample


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
            f"{arg_name} must be an instance of {expected_type}. Got {type(value)}"
        )


class RandomVariable(metaclass=ABCMeta):
    """
    Abstract base class for latent and observed random variables.
    """

    def __init__(self, **kwargs):
        """
        Default constructor
        """
        pass

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
            Additional keyword arguments passed through to internal
            `sample` calls, should there be any.

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
        Alias for `sample`.
        """
        return self.sample(**kwargs)


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
            Additional keyword arguments passed through to internal
            `sample` calls, should there be any.

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
            Additional keyword arguments passed through to
            internal `sample` calls, should there be any.

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
            Dictionary of arguments passed to the
            [`numpyro.infer.hmc.NUTS`][] constructor.
            Default None.
        mcmc_args : dict, optional
            Dictionary of arguments passed to the
            [`numpyro.infer.mcmc.MCMC`][] constructor.
            Default None.

        Returns
        -------
        None
        """

        if nuts_args is None:
            nuts_args = dict()

        if "find_heuristic_step_size" not in nuts_args:
            nuts_args["find_heuristic_step_size"] = True

        if "init_strategy" not in nuts_args:
            nuts_args["init_strategy"] = init_to_sample

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
            Dictionary of arguments passed to the kernel
            [`numpyro.infer.hmc.NUTS`][] constructor.
            Defaults to None.
        mcmc_args : dict, optional
            Dictionary of arguments passed to the MCMC runner
            [`numpyro.infer.mcmc.MCMC`][] constructor.
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
            rand_int = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max)
            rng_key = jr.key(rand_int)

        self.mcmc.run(rng_key=rng_key, **kwargs)

        return None

    def print_summary(
        self,
        prob: float = 0.9,
        exclude_deterministic: bool = True,
    ) -> None:
        """
        A wrapper of [`numpyro.infer.mcmc.MCMC.print_summary`][].

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

    def posterior_predictive(
        self,
        rng_key: ArrayLike | None = None,
        numpyro_predictive_args: dict = {},
        **kwargs,
    ) -> dict:
        """
        A wrapper of [`numpyro.infer.util.Predictive`][] to generate
        posterior predictive samples.

        Parameters
        ----------
        rng_key : ArrayLike, optional
            Random key for the Predictive function call. Defaults to None.
        numpyro_predictive_args : dict, optional
            Dictionary of arguments to be passed to the
            [`numpyro.infer.util.Predictive`][] constructor.
        **kwargs
            Additional named arguments passed to the
            `__call__()` method of
            [`numpyro.infer.util.Predictive`][].

        Returns
        -------
        dict
        """
        if self.mcmc is None:
            raise ValueError(
                "No posterior samples available. Run model with model.run()."
            )

        if rng_key is None:
            rand_int = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max)
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
        A wrapper for [`numpyro.infer.util.Predictive`][]
        to generate prior predictive samples.

        Parameters
        ----------
        rng_key : ArrayLike, optional
            Random key for the Predictive function call.
            Default None.
        numpyro_predictive_args : dict, optional
            Dictionary of arguments to be passed to
            the [`numpyro.infer.util.Predictive`][]
            constructor. Default None.
        **kwargs
            Additional named arguments passed to the
            `__call__()` method of
            [`numpyro.infer.util.Predictive`][].

        Returns
        -------
        dict
        """

        if rng_key is None:
            rand_int = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max)
            rng_key = jr.key(rand_int)

        predictive = Predictive(
            model=self.model,
            posterior_samples=None,
            **numpyro_predictive_args,
        )

        return predictive(rng_key, **kwargs)
