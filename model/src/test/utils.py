# -*- coding: utf-8 -*-

"""
test utilities
"""

import numpyro.distributions as dist
import pyrenew.transformation as t
from pyrenew.metaclass import (
    DistributionalRV,
    RandomVariable,
    SampledValue,
    TransformedRandomVariable,
)
from pyrenew.process import RandomWalk


class SimpleRt(RandomVariable):
    """
    Helper class to create a default Rt
    RandomVariable for testing.
    """

    def __init__(self, name: str = "Rt_rv"):
        """
        Default constructor

        Parameters
        -----------
        name : str
           Name assigned to the RandomVariable.
           If None, then defaults to "Rt_rv"

        Returns
        -------
        None
        """
        self.name = name
        self.rt_rv_ = TransformedRandomVariable(
            name=name + "log_rt_random_walk",
            base_rv=RandomWalk(
                name="log_rt",
                step_rv=DistributionalRV(
                    name="rw_step_rv", distribution=dist.Normal(0, 0.025)
                ),
            ),
            transforms=t.ExpTransform(),
        )
        self.rt_init_rv_ = DistributionalRV(
            name=name + "init_log_rt", distribution=dist.Normal(0, 0.2)
        )

    def sample(self, n=None, **kwargs) -> SampledValue:
        """
        Sample method

        Returns
        -------
        SampledValue
        """
        init_rt, *_ = self.rt_init_rv_.sample()
        return self.rt_rv_(init_vals=init_rt.value, n=n)

    @staticmethod
    def validate(self):
        """
        No validation.
        """
        pass
