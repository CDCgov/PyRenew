#!/usr/bin/env/python
# -*- coding: utf-8 -*-

"""
Observation helper classes
"""

import numpyro
import numpyro.distributions as dist
from pyrenew.metaclasses import RandomProcess


class NegativeBinomialObservation(RandomProcess):
    def __init__(
        self,
        concentration_prior: dist.Distribution,
        concentration_suffix: str = "_concentration",
        parameter_name="negbinom_rv",
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
        self.parameter_name = parameter_name
        self.concentration_suffix = concentration_suffix

    def sample(self, predicted_value, data=None, obs=None):
        concentration_parameter = numpyro.sample(
            self.parameter_name + self.concentration_suffix,
            self.concentration_prior,
        )

        return numpyro.sample(
            self.parameter_name,
            dist.NegativeBinomial2(
                mean=predicted_value, concentration=concentration_parameter
            ),
            obs=obs,
        )

    @staticmethod
    def validate():
        return None
