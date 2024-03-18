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
        mean_varname="mean",
        counts_varname="counts",
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
        :param mean_varname: Name of the element in `random_variables` that will
            hold the rate when calling `PoissonObservation.sample()`. Defaults
            to 'mean'.
        :type mean_varname: str, optional
        :param counts_varname: Name of the element in `random_variables` that will
            hold the observed count when calling `PoissonObservation.sample()`.
            Defaults to 'counts'.
        :type counts_varname: str, optional
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
        self.mean_varname = mean_varname
        self.counts_varname = counts_varname

    def sample(
        self,
        random_variables: dict,
        constants: dict = None,
    ):
        """Sample from the negative binomial distribution

        :param random_variables: A dictionary containing the `mean` parameter,
            and possibly containing `counts`, which is passed to `obs`
            `numpyro.sample()`.
        :type random_variables: dict, optional
        :param constants: Ignored, defaults to dict().
        :type constants: dict, optional
        :return: _description_
        :rtype: _type_
        """
        return numpyro.sample(
            self.parameter_name,
            dist.NegativeBinomial2(
                mean=random_variables.get(self.mean_varname),
                concentration=self.sample_prior(),
            ),
            obs=random_variables.get(self.counts_varname, None),
        )

    @staticmethod
    def validate(concentration_prior) -> None:
        assert isinstance(
            concentration_prior, (dist.Distribution, nums.Number)
        )
        return None
