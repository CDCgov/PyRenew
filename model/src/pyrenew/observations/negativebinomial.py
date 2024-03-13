#!/usr/bin/env/python
# -*- coding: utf-8 -*-

"""
Observation helper classes
"""

from pyrenew.metaclasses import RandomProcess

import numpyro
import numpyro.distributions as dist
from numpy.typing import ArrayLike

class NegativeBinomialObservation(RandomProcess):
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

    def validate(self):
        return None