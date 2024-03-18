#!/usr/bin/env/python
# -*- coding: utf-8 -*-

"""
Observation helper classes
"""

from abc import ABCMeta, abstractmethod
from collections import namedtuple

import jax
from numpyro.infer import MCMC, NUTS
from pyrenew.mcmcutils import spread_draws


def _assert_sample_and_rtype(
    fn: "RandomProcess", skip_if_none: bool = True
) -> None:
    """Return type-checking for RandomProcess's sample function

    Objects passed as `RandomProcess` should (a) have a `sample()` method that
    (b) returns either a tuple or a named tuple.

    :param fn: _description_
    :type fn: RandomProcess
    :raises Exception: _description_
    :raises Exception: _description_
    :return: _description_
    :rtype: _type_
    """

    if (fn is None) and (not skip_if_none):
        Exception(
            "The passed object cannot be None. It should be RandomProcess"
        )

    if not isinstance(fn, RandomProcess):
        raise Exception(f"{fn.__name__} is not an instance of RandomProcess.")

    try:
        sfun = fn.sample
    except Exception:
        raise Exception(
            f"The RandomProcess {fn.__name__} does not have a sample function."
        )  # noqa: E722

    # Getting the return annotation (if any)
    rettype = sfun.__annotations__.get("return", None)

    if rettype is None:
        raise Exception(
            f"The RandomProcess {fn.__name__} does not have return type "
            + "annotation."
        )

    if not isinstance(rettype(), (tuple, namedtuple)):
        raise Exception(
            f"The RandomProcess {fn.__name__}'s return type annotation is not"
            + "a tuple"
        )

    return None


class RandomProcess(metaclass=ABCMeta):
    """
    Abstract base class for an observation
    with a single predicted value and optional
    auxiliary parameters governing properties
    such as observation noise
    """

    def __init__(self, **kwargs):
        """
        Default constructor
        """
        pass

    @abstractmethod
    def sample(self, random_variables: dict = None, constants: dict = None):
        """Sample method of the process

        The method desing in the class should have two dictionaries:
        `random_variables` and `constants`.

        - `randon_variables`: This dictionary contains any data that sample
          function may pass to `numpyro.sample(obs=...)`.

        - `constants`: Contains any data that the sample function does not pass
          to `numpyro.sample(obs=...)`.

        :param random_variables: _description_, defaults to None
        :type random_variables: dict, optional
        :param constants: _description_, defaults to None
        :type constants: dict, optional
        """
        pass

    @staticmethod
    @abstractmethod
    def validate():
        pass


class Model(metaclass=ABCMeta):
    # Since initialized in none, values not shared across instances
    kernel = None
    mcmc = None

    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def validate():
        pass

    @abstractmethod
    def model(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ):
        pass

    def _init_model(
        self,
        num_warmup,
        num_samples,
        nuts_args: dict = None,
        mcmc_args: dict = None,
    ) -> None:
        """Creates the NUTS kernel and MCMC model

        Args:
            nuts_args (dict, optional): _description_. Defaults to None.
            mcmc_args (dict, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
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
        rng_key: jax.random.PRNGKey = jax.random.PRNGKey(54),
        random_variables: dict = None,
        constants: dict = None,
        nuts_args: dict = None,
        mcmc_args: dict = None,
    ) -> None:
        """Runs the model

        Args:
            nuts_args (dict, optional): _description_. Defaults to None.
            mcmc_args (dict, optional): _description_. Defaults to None.

        Returns:
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
        """A wrapper of MCMC.print_summary.

        Args:
            prob (float, optional): _description_. Defaults to 0.9.
            exclude_deterministic (bool, optional): _description_. Defaults to True.
        """
        return self.mcmc.print_summary(prob, exclude_deterministic)

    def plot(self):
        pass

    def spread_draws(self, variables_names):
        return spread_draws(self.mcmc.get_samples(), variables_names)
