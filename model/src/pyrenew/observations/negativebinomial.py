#!/usr/bin/env/python
# -*- coding: utf-8 -*-


import numbers as nums

import numpyro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from pyrenew.metaclasses import RandomProcess


class NegativeBinomialObservation(RandomProcess):
    """Negative Binomial observation"""

    def __init__(
        self,
        concentration_prior: dist.Distribution | ArrayLike,
        concentration_suffix: str = "_concentration",
        parameter_name="negbinom_rv",
    ):
        """Default constructor

        :param concentration_prior: dist.Distribution
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
        :type concentration_prior: dist.Distribution
        :param concentration_suffix: _description_, defaults to "_concentration"
        :type concentration_suffix: str, optional
        :param parameter_name: _description_, defaults to "negbinom_rv"
        :type parameter_name: str, optional
        """

        self.validate(concentration_prior)

        if isinstance(concentration_prior, dist.Distribution):
            self.sample_prior = lambda: numpyro.sample(
                self.parameter_name + self.concentration_suffix,
                concentration_prior,
            )
        else:
            self.sample_prior = lambda: concentration_prior

        self.parameter_name = parameter_name
        self.concentration_suffix = concentration_suffix

    def sample(
        self, predicted_value: ArrayLike, data: dict = dict(), obs=None
    ):
        """Sample from the negative binomial distribution

        :param predicted_value: Mean parameter of the negative binomial
        :type predicted_value: ArrayLike
        :param data: Ignored, defaults to dict()
        :type data: dict, optional
        :param obs: _description_, defaults to None
        :type obs: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        return numpyro.sample(
            self.parameter_name,
            dist.NegativeBinomial2(
                mean=predicted_value, concentration=self.sample_prior()
            ),
            obs=obs,
        )

    @staticmethod
    def validate(concentration_prior) -> None:
        assert isinstance(
            concentration_prior, (dist.Distribution, nums.Number)
        )
        return None
