#!/usr/bin/env/python
# -*- coding: utf-8 -*-

"""
Observation helper classes
"""

from abc import ABCMeta, abstractmethod

import numpyro
import numpyro.distributions as dist
from numpy.typing import ArrayLike


class Observation(metaclass=ABCMeta):
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


class PoissonObservation(Observation):
    """
    Poisson observation process
    """

    def sample(self, parameter_name, predicted_value, data=None, obs=None):
        return numpyro.sample(
            parameter_name, dist.Poisson(rate=predicted_value), obs=obs
        )


class NegativeBinomialObservation(Observation):
    def __init__(
        self,
        concentration_prior: dist.Distribution,
        concentration_suffix: str = "_concentration",
    ):
        """
        Default constructor

        Parameters
        ----------
        concentration_prior : dist.Distribution
            Numpyro distribution from which to
            sample the positive concentration
            parameter of the negative binomial.
            This parameter is sometimes called
            k, phi, or the "dispersion"
            or "overdispersion" parameter,
            despite the fact that larger values
            imply that the distribution becomes
            more Poissonian, while smaller ones
            imply a greater degree of dispersion.
        """
        self.concentration_prior = concentration_prior
        self.concentration_suffix = concentration_suffix

    def sample(self, parameter_name, predicted_value, data=None, obs=None):
        concentration_parameter = numpyro.sample(
            parameter_name + self.concentration_suffix,
            self.concentration_prior,
        )

        return numpyro.sample(
            parameter_name,
            dist.NegativeBinomial2(
                mean=predicted_value, concentration=concentration_parameter
            ),
            obs=obs,
        )
