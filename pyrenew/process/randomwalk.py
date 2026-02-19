# numpydoc ignore=GL08

from typing import Any

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
        name: str,
        step_rv: RandomVariable,
        **kwargs: Any,
    ) -> None:
        """
        Default constructor.

        Parameters
        ----------
        name
            A name for this random variable.
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
            name=name,
            fundamental_process=IIDRandomSequence(
                name=f"{name}_iid_seq", element_rv=step_rv
            ),
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
        name: str,
        **kwargs: Any,
    ) -> None:
        """
        Default constructor.

        Parameters
        ----------
        name
            A name for this random variable.
            The internal step distribution is named
            ``f"{name}_step"``.
        **kwargs
            Additional keyword arguments passed
            to the parent class constructor.

        Returns
        -------
        None
        """
        super().__init__(
            name=name,
            step_rv=DistributionalVariable(
                name=f"{name}_step", distribution=dist.Normal(0.0, 1.0)
            ),
            **kwargs,
        )
