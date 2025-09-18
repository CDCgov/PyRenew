# numpydoc ignore=GL08

import numpyro.distributions as dist

from pyrenew.metaclass import RandomVariable
from pyrenew.process.differencedprocess import DifferencedProcess
from pyrenew.process.iidrandomsequence import IIDRandomSequence
from pyrenew.randomvariable import DistributionalVariable


class RandomWalk(DifferencedProcess):
    """
    Class for a Markovian
    random walk with an arbitrary
    step distribution, implemented
    via DifferencedProcess and
    IIDRandomSequence
    """

    def __init__(
        self,
        step_rv: RandomVariable,
        **kwargs,
    ):
        """
        Default constructor

        Parameters
        ----------
        step_rv
            RandomVariable representing a single step
            (difference) in the random walk.

        **kwargs
            Additional keyword arguments passed to the parent
            class constructor.

        Returns
        -------
        None
        """
        super().__init__(
            fundamental_process=IIDRandomSequence(element_rv=step_rv),
            differencing_order=1,
            **kwargs,
        )


class StandardNormalRandomWalk(RandomWalk):
    """
    A random walk with standard Normal
    (mean = 0, standard deviation = 1)
    steps, implmemented via the base
    RandomWalk class.
    """

    def __init__(
        self,
        step_rv_name: str,
        **kwargs,
    ):
        """
        Default constructor
        Parameters
        ----------
        step_rv_name
            Name for the DistributionalVariable
            from which the Normal(0, 1)
            steps are sampled.
        **kwargs
            Additional keyword arguments passed
            to the parent class constructor.
        Return
        ------
        None
        """
        super().__init__(
            step_rv=DistributionalVariable(
                name=step_rv_name, distribution=dist.Normal(0.0, 1.0)
            ),
            **kwargs,
        )
