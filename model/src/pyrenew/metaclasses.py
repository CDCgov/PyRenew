#!/usr/bin/env/python
# -*- coding: utf-8 -*-

"""
Observation helper classes
"""

from abc import ABCMeta, abstractmethod

import jax
from numpyro.infer import MCMC, NUTS
from pyrenew.mcmcutils import spread_draws


class RandomProcess(metaclass=ABCMeta):
    """
    Abstract base class for an observation
    with a single predicted value and optional
    auxiliary parameters governing properties
    such as observation noise
    """

    def __init__(self):
        """
        Default constructor
        """
        pass

    @abstractmethod
    def sample(self, obs: dict = dict(), data: dict = dict(), **kargs):
        """Sample method of the process

        The method desing in the class should have `obs` and `data` by default.
        The observed data (`obs`) should be contained in a dictionary. The
        `data` argument should be used for additional paramters.

        :param obs: _description_, defaults to None
        :type obs: _type_, optional
        :param data: _description_, defaults to None
        :type data: _type_, optional
        """
        pass

    @staticmethod
    @abstractmethod
    def validate():
        pass


class Model(metaclass=ABCMeta):
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
    def model(self, data):
        pass

    def _init_model(
        self,
        num_warmup,
        num_samples,
        nuts_args: dict = dict(),
        mcmc_args: dict = dict(),
    ) -> None:
        """Creates the NUTS kernel and MCMC model

        Args:
            nuts_args (dict, optional): _description_. Defaults to None.
            mcmc_args (dict, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

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
        obs: dict = dict(),
        data: dict = dict(),
        nuts_args: dict = dict(),
        mcmc_args: dict = dict(),
    ) -> None:
        """Runs the model

        Args:
            nuts_args (dict, optional): _description_. Defaults to None.
            mcmc_args (dict, optional): _description_. Defaults to None.

        Returns:
            None
        """

        if self.mcmc is None:
            self._init_model(
                num_warmup=num_warmup,
                num_samples=num_samples,
                nuts_args=nuts_args,
                mcmc_args=mcmc_args,
            )

        self.mcmc.run(rng_key=rng_key, obs=obs, data=data)

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
