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
        self.parameter_name = parameter_name
        return None

    def sample(self, predicted_value, data=None, obs=None):
        return numpyro.sample(
            self.parameter_name, dist.Poisson(rate=predicted_value), obs=obs
        )

    @staticmethod
    def validate():
        None
