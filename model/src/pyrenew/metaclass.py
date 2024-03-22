# -*- coding: utf-8 -*-

"""
pyrenew helper classes
"""

from abc import ABCMeta, abstractmethod

import jax
import polars as pl
from numpyro.infer import MCMC, NUTS
from pyrenew.mcmcutils import spread_draws


def _assert_sample_and_rtype(
    rp: "RandomVariable", skip_if_none: bool = True
) -> None:
    """Return type-checking for RandomVariable's sample function

    Objects passed as `RandomVariable` should (a) have a `sample()` method that
    (b) returns either a tuple or a named tuple.

    Parameters
    ----------
    rp : RandomVariable
        Random variable to check.
    skip_if_none: bool
        When `True` it returns if `rp` is None.

    Returns
    -------
    None
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
    rettype = sfun.__annotations__.get("return", None)

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
    """

    def __init__(self, **kwargs):
        """
        Default constructor
        """
        pass

    @abstractmethod
    def sample(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        """Sample method of the process

        The method desing in the class should have two dictionaries:
        `random_variables` and `constants`.

        - `randon_variables`: This dictionary contains any data that sample
          function may pass to `numpyro.sample(obs=...)`.

        - `constants`: Contains any data that the sample function does not pass
          to `numpyro.sample(obs=...)`.

        Parameters
        ----------
        random_variables : dict, optional
            Dictionary of random variables, defaults to None
        constants : dict, optional
            Dictionary of constants

        Returns
        -------
        tuple
        """
        pass

    @staticmethod
    @abstractmethod
    def validate(**kwargs) -> None:
        pass


class Model(metaclass=ABCMeta):
    # Since initialized in none, values not shared across instances
    kernel = None
    mcmc = None

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @staticmethod
    @abstractmethod
    def validate() -> None:
        pass

    @abstractmethod
    def sample(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        pass

    def _init_model(
        self,
        num_warmup,
        num_samples,
        nuts_args: dict = None,
        mcmc_args: dict = None,
    ) -> None:
        """Creates the NUTS kernel and MCMC model

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
            model=self.sample,
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
        rng_key: jax.random.PRNGKey = jax.random.PRNGKey(54),
        random_variables: dict = None,
        constants: dict = None,
        nuts_args: dict = None,
        mcmc_args: dict = None,
    ) -> None:
        """Runs the model

        Parameters
        ----------
        nuts_args : dict, optional
            Dictionary of arguments passed to the NUTS. Defaults to None.
        mcmc_args : dict, optional
            Dictionary of passed to the MCMC sampler. Defaults to None.

        Returns
        -------
            None
        """

        if random_variables is None:
            random_variables = dict()

        if constants is None:
            constants = dict()

        if self.mcmc is None:
            self._init_model(
                num_warmup=num_warmup,
                num_samples=num_samples,
                nuts_args=nuts_args,
                mcmc_args=mcmc_args,
            )

        self.mcmc.run(
            rng_key=rng_key,
            random_variables=random_variables,
            constants=constants,
        )

        return None

    def print_summary(
        self,
        prob: float = 0.9,
        exclude_deterministic: bool = True,
    ) -> None:
        """A wrapper of MCMC.print_summary"""
        return self.mcmc.print_summary(prob, exclude_deterministic)

    def spread_draws(self, variables_names: list) -> pl.DataFrame:
        """A wrapper of mcmcutils.spread_draws"""
        return spread_draws(self.mcmc.get_samples(), variables_names)
