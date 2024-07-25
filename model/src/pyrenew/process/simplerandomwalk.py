# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
from numpyro.contrib.control_flow import scan
from pyrenew.metaclass import RandomVariable, SampledValue


class SimpleRandomWalkProcess(RandomVariable):
    """
    Class for a Markovian
    random walk with an a
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
            name sites within it in :fun :`numpyro.sample()`
            calls.
        step_rv : RandomVariable
            RandomVariable representing the step distribution.
        init_rv : RandomVariable
            RandomVariable representing the initial value of
            the process
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
        self.init_rv = init_rv
        self.t_start = t_start
        self.t_unit = t_unit

    def sample(
        self,
        n_steps: int,
        **kwargs,
    ) -> tuple:
        """
        Sample from the random walk.

        Parameters
        ----------
        n_steps : int
            Length of the walk to sample.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        tuple
            With a single array of shape (n_steps,).
        """

        init, *_ = self.init_rv(**kwargs)

        def transition(x_prev, _):
            # numpydoc ignore=GL08
            diff, *_ = self.step_rv(**kwargs)
            x_curr = x_prev + diff.value
            return x_curr, x_curr

        _, x = scan(
            transition,
            init=init.value,
            xs=jnp.arange(n_steps - 1),
        )

        return (
            SampledValue(
                jnp.hstack([init.value, x.flatten()]),
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
