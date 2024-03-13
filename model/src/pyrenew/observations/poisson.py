#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from pyrenew.metaclasses import RandomProcess

import numpyro
import numpyro.distributions as dist

class PoissonObservation(RandomProcess):
    """
    Poisson observation process
    """

    def sample(self, parameter_name, predicted_value, data=None, obs=None):
        return numpyro.sample(
            parameter_name, dist.Poisson(rate=predicted_value), obs=obs
        )
    
    def validate(self):
        None


