# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import numpyro.distributions as dist

from pyrenew.metaclass import DistributionalRV, RandomVariable
from pyrenew.process.differencedprocess import DifferencedProcess
from pyrenew.process.iidrandomsequence import IIDRandomSequence


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
        step_sequence_suffix="_steps",
        **kwargs,
    ):
        """
        Default constructor

        Parameters
        ----------
        name : str
            A name for the random variable, used to
            name sites within it in :fun:`numpyro.sample()`
            calls.

        step_rv : RandomVariable
            RandomVariable representing a single step
            (difference) in the random walk.

        step_sequence_suffix : str
            Suffix to append to the RandomVariable's
            name when naming the IIDRandomSequence that
            holds its steps (differences). Default
            "_steps".

        **kwargs :
            Additional keyword arguments passed to the parent
            class constructor.

        Returns
        -------
        None
        """
        super().__init__(
            name=name,
            fundamental_process=IIDRandomSequence(
                name=name + step_sequence_suffix, element_rv=step_rv
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
        step_rv_suffix="_step",
        step_sequence_suffix="_steps",
        **kwargs,
    ):
        """
        Default constructor
        Parameters
        ----------
        name : str
            A name for the random variable.
        step_rv_suffix :
            Suffix to append to the random variable
            when naming the DistributionalRV
            from which its Normal(0, 1)
            steps are sampled. Default "_step".
        step_sequence_suffix : str
            See :class:`RandomWalk`. Default "_steps".
        **kwargs:
            Additional keyword arguments passed
            to the parent class constructor.
        Return
        ------
        None
        """
        super().__init__(
            name=name,
            step_rv=DistributionalRV(
                name=name + step_rv_suffix, distribution=dist.Normal(0.0, 1.0)
            ),
            step_sequence_suffix=step_sequence_suffix,
            **kwargs,
        )
