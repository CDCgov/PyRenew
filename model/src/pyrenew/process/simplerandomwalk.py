# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from pyrenew.metaclass import RandomVariable


class SimpleRandomWalkProcess(RandomVariable):
    """
    Class for a Markovian
    random walk with an a
    arbitrary step distribution
    """

    def __init__(
        self,
        error_distribution: dist.Distribution,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        error_distribution : dist.Distribution
            Passed to numpyro.sample.

        Returns
        -------
        None
        """
        self.error_distribution = error_distribution

    def sample(
        self,
        duration: int,
        name: str = "randomwalk",
        init: float = None,
        **kwargs,
    ) -> tuple:
        """
        Samples from the randomwalk

        Parameters
        ----------
        duration : int
            Length of the walk.
        name : str, optional
            Passed to numpyro.sample, by default "randomwalk"
        init : float, optional
            Initial point of the walk, by default None
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        tuple
        """

        if init is None:
            init = npro.sample(name + "_init", self.error_distribution)
        diffs = npro.sample(
            name + "_diffs", self.error_distribution.expand((duration,))
        )

        return (init + jnp.cumsum(jnp.pad(diffs, [1, 0], constant_values=0)),)

    @staticmethod
    def validate():
        """
        Validates inputted parameters, implementation pending.
        """
        return None
