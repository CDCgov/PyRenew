# -*- coding: utf-8 -*-

"""
test utilities
"""

import numpyro.distributions as dist
import pyrenew.transformation as t
from pyrenew.metaclass import DistributionalRV, TransformedRandomVariable
from pyrenew.process import SimpleRandomWalkProcess


def get_default_rt():
    """
    Helper function to create a default Rt
    RandomVariable for testing.

    Returns
    -------
    TransformedRandomVariable :
       A log-scale random walk with fixed
       init value and step size priors
    """
    return TransformedRandomVariable(
        "Rt_rv",
        base_rv=SimpleRandomWalkProcess(
            name="log_rt",
            step_rv=DistributionalRV(
                name="rw_step_rv", dist=dist.Normal(0, 0.025)
            ),
            init_rv=DistributionalRV(
                name="init_log_rt", dist=dist.Normal(0, 0.2)
            ),
        ),
        transforms=t.ExpTransform(),
    )
