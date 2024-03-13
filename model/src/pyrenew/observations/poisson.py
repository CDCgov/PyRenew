#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import numpyro
import numpyro.distributions as dist
from pyrenew.metaclasses import RandomProcess


class PoissonObservation(RandomProcess):
    """
    Poisson observation process
    """

    def sample(self, parameter_name, predicted_value, data=None, obs=None):
        return numpyro.sample(
            parameter_name, dist.Poisson(rate=predicted_value), obs=obs
        )

    @staticmethod
    def validate():
        None
