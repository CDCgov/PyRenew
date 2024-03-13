#!/usr/bin/env/python
# -*- coding: utf-8 -*-

"""
Observation helper classes
"""

from abc import ABCMeta, abstractmethod
from numpy.typing import ArrayLike
import numpyro as npro
from numpyro.infer import MCMC, NUTS
import jax


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
    def sample(
        self, parameter_name, predicted_value: ArrayLike, data=None, obs=None
    ):
        """
        Sampling method that concrete
        versions should implement.
        """
        pass

    @abstractmethod
    def validate(self):
        pass

class Model(metaclass=ABCMeta):

    kernel = None
    mcmc = None
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def model(self, data):
        pass

    def _init_model(
            self,
            num_warmup,
            num_samples,
            nuts_args : dict = dict(),
            mcmc_args : dict = dict(),
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
        rng_key : jax.random.PRNGKey = jax.random.PRNGKey(54),
        data=None,
        nuts_args : dict = dict(),
        mcmc_args : dict = dict(),
        ) -> None:
        """Runs the model

        Args:
            nuts_args (dict, optional): _description_. Defaults to None.
            mcmc_args (dict, optional): _description_. Defaults to None.

        Returns:
            None
        """

        if (self.mcmc == None):
            self._init_model(
                num_warmup=num_warmup,
                num_samples=num_samples,
                nuts_args=nuts_args,
                mcmc_args=mcmc_args,
                )
        
        self.mcmc.run(rng_key=rng_key, data=data)

        return None

    def print_summary(self, kargs=dict()):
        self.mcmc.print_summary(**kargs)

    def plot(self):
        pass