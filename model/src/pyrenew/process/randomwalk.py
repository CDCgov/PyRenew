# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import numpyro.distributions as dist
from pyrenew.process.differencedprocess import DifferencedProcess
from pyrenew.process.iidrandomsequence import IIDRamdomSequence


class RandomWalk(DifferencedProcess):
    """
    Class for a Markovian
    random walk with an arbitrary
    step distribution, implemented
    via DifferencedProcess and IIDRamdomSequence
    """

    def __init__(
        self,
        name: str,
        step_distribution: dist.Distribution,
        step_suffix="_step",
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

        step_distribution : dist.Distribution
            numpyro.distributions.Distribution
            representing the step distribution
            of the random walk.

        step_suffix : str
            Suffix to append to the RandomVariable's
            name when naming the random variable that
            holds its steps (differences). Default
            "_step".

        **kwargs :
            Additional keyword arguments passed to the parent
            class constructor.

        Returns
        -------
        None
        """
        super().__init__(
            name=name,
            fundamental_process=IIDRamdomSequence(
                name=name + step_suffix, distribution=step_distribution
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

    def __init__(self, name: str, step_suffix="_step", **kwargs):
        """
        Default constructor
        Parameters
        ----------
        name : str
            A name for the random variable.
        step_suffix : str
            See :class:`RandomWalk`
        **kwargs:
            Additional keyword arguments passed
            to the parent class constructor.
        Return
        ------
        None
        """
        super().__init__(
            name=name, step_distribution=dist.Normal(0, 1), **kwargs
        )
