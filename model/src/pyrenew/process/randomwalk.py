# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from pyrenew.metaclass import DistributionalRV, RandomVariable, SampledValue


class RandomWalk(RandomVariable):
    """
    Class for a Markovian
    random walk with an arbitrary
    step distribution
    """

    def __init__(
        self,
        name: str,
        step_rv: RandomVariable,
        init_rv: RandomVariable,
        t_start: int = None,
        t_unit: int = None,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name : str
            A name for the random variable, used to
            name sites within it in :fun:`numpyro.sample()`
            calls.
        step_rv : RandomVariable
            RandomVariable representing the step distribution.
        t_start : int
            See :class:`RandomVariable`
        t_unit : int
            See :class:`RandomVariable`

        Returns
        -------
        None
        """
        self.name = name
        self.step_rv = step_rv
        self.t_start = t_start
        self.t_unit = t_unit

    def sample(
        self,
        n_steps: int,
        init_val: float,
        **kwargs,
    ) -> tuple:
        """
        Sample from the random walk.

        Parameters
        ----------
        n_steps : int
            Length of the walk to sample.
        init_val : float
            Initial value of the walk.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        tuple
            With a single array of shape (n_steps,).
        """

        def transition(x_prev, _):
            # numpydoc ignore=GL08
            diff, *_ = self.step_rv(**kwargs)
            x_curr = x_prev + diff.value
            return x_curr, x_curr

        _, x = scan(
            transition,
            init=init_val,
            xs=jnp.arange(n_steps - 1),
        )

        return (
            SampledValue(
                jnp.hstack([init_val, x.flatten()]),
                t_start=self.t_start,
                t_unit=self.t_unit,
            ),
        )

    @staticmethod
    def validate():
        """
        Validates input parameters, implementation pending.
        """
        super().validate()
        return None


class StandardNormal(RandomWalk):
    """
    A random walk with
    standard Normal (mean = 0, standard deviation = 1)
    steps, implmemented via the base RandomWalk class.
    """

    def __init__(self, name: str, step_suffix="_step", **kwargs):
        """
        Default constructor

        Parameters
        ----------
        name : str
            A name for the random variable.

        step_suffix : str
           A suffix to append to the name when
           sampling the random walk steps via
           numpyro.sample (the name of the site/
           parameter in numpyro will be
           self.name + self.step_suffix.

        **kwargs:
            Additional keyword arguments passed
            to the parent class constructor.
        """
        super().__init__(
            name=name,
            step_rv=DistributionalRV(
                name=name + step_suffix, dist=dist.Normal(loc=0, scale=1)
            ),
            **kwargs,
        )
