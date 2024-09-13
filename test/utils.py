"""
test utilities
"""

import numpyro.distributions as dist
from jax.typing import ArrayLike

import pyrenew.transformation as t
from pyrenew.metaclass import RandomVariable
from pyrenew.process import RandomWalk
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable


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
        name = "Rt_rv"
        self.rt_rv_ = TransformedVariable(
            name=f"{name}_log_rt_random_walk",
            base_rv=RandomWalk(
                name="log_rt",
                step_rv=DistributionalVariable(
                    name="rw_step_rv", distribution=dist.Normal(0, 0.025)
                ),
            ),
            transforms=t.ExpTransform(),
        )
        self.rt_init_rv_ = DistributionalVariable(
            name=f"{name}_init_log_rt", distribution=dist.Normal(0, 0.2)
        )

    def sample(self, n=None, **kwargs) -> ArrayLike:
        """
        Sample method

        Returns
        -------
        ArrayLike
        """
        init_rt = self.rt_init_rv_()
        return self.rt_rv_(init_vals=init_rt, n=n)

    @staticmethod
    def validate(self):
        """
        No validation.
        """
        pass
