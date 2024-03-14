#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import numpyro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
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

    def sample(self, predicted_value: ArrayLike, data: dict = None, obs=None):
        """Sample from the Poisson process

        :param predicted_value: Rate parameter passed to
            numpyro.distributions.Poisson.
        :type predicted_value: ArrayLike
        :param data: Ignored, defaults to None
        :type data: _type_, optional
        :param obs: Observed data passed to numpyro.sample, defaults to None
        :type obs: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        return numpyro.sample(
            self.parameter_name, dist.Poisson(rate=predicted_value), obs=obs
        )

    @staticmethod
    def validate():
        None
