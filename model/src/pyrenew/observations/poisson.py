#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import numpyro
import numpyro.distributions as dist
from pyrenew.metaclasses import RandomProcess


class PoissonObservation(RandomProcess):
    """
    Poisson observation process
    """

    def __init__(self, parameter_name="poisson_rv") -> None:
        """Constructor

        :param parameter_name: Passed to numpyro.sample, defaults to
            "poisson_rv"
        :type parameter_name: str, optional
        :return: _description_
        :rtype: _type_
        """

        self.parameter_name = parameter_name
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
            dist.Poisson(rate=random_variables.get("rate")),
            obs=random_variables.get("counts", None),
        )

    @staticmethod
    def validate():
        None
