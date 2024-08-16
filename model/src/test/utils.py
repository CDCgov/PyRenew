# -*- coding: utf-8 -*-

"""
test utilities
"""

import numpyro.distributions as dist
import pyrenew.transformation as t
from pyrenew.metaclass import DistributionalRV, TransformedRandomVariable
from pyrenew.process import RandomWalk


def simple_rt(arg_name: str = "Rt_rv"):
    """
    Helper function to create a default Rt
    RandomVariable for testing.

    Parameters
    -----------
    arg_name : str
        Name assigned to the randonvariable.
        If None, then defaults to "Rt_rv"

    Returns
    -------
    TransformedRandomVariable :
       A log-scale random walk with fixed
       init value and step size priors
    """
    return TransformedRandomVariable(
        arg_name,
        base_rv=RandomWalk(
            name="log_rt",
            step_distribution=dist.Normal(0, 0.025),
            init_rv=DistributionalRV(
                name="init_log_rt", distribution=dist.Normal(0, 0.2)
            ),
        ),
        transforms=t.ExpTransform(),
    )
