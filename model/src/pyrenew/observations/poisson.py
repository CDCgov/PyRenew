#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import numpyro
import numpyro.distributions as dist
from pyrenew.metaclasses import RandomProcess


class PoissonObservation(RandomProcess):
    """
    Poisson observation process
    """

    def __init__(
        self,
        parameter_name="poisson_rv",
        rate_varname="rate",
        counts_varname="counts",
        eps=1e-8,
    ) -> None:
        """Default Constructor

        :param parameter_name: Passed to numpyro.sample, defaults to
            "poisson_rv"
        :type parameter_name: str, optional
        :param rate_varname: Name of the element in `random_variables` that will
            hold the rate when calling `PoissonObservation.sample()`. Defaults
            to 'rate'.
        :type rate_varname: str, optional
        :param counts_varname: Name of the element in `random_variables` that will
            hold the observed count when calling `PoissonObservation.sample()`.
            Defaults to 'counts'.
        :type counts_varname: str, optional
        :return: _description_
        :rtype: _type_
        """

        self.parameter_name = parameter_name
        self.rate_varname = rate_varname
        self.counts_varname = counts_varname
        self.eps = eps

        return None

    def sample(
        self,
        random_variables: dict,
        constants: dict = None,
    ):
        """Sample from the Poisson process

        :param random_variables: A dictionary containing the rate parameter
            passed to `numpyro.distributions.Poisson()`, and possible containing
            `counts` passed to `obs` in `numpyro.sample()`.
        :type random_variables: _type_, optional
        :param constants: Ignored, defaults to None
        :type constants: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        return numpyro.sample(
            self.parameter_name,
            dist.Poisson(
                rate=random_variables.get(self.rate_varname) + self.eps
            ),
            obs=random_variables.get(self.counts_varname, None),
        )

    @staticmethod
    def validate():
        None
