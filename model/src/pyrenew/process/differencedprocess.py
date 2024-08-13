# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable, SampledValue


class DifferencedProcess(RandomVariable):
    """
    Class for differenced stochastic process X(t),
    constructed by placing a fundamental stochastic
    process on the first differences (rates of change).
    """

    def __init__(
        self,
        name: str,
        fundamental_process: RandomVariable,
        differencing_order: int,
        **kwargs,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name : str
            Name of the stochastic process
        fundamental_process : RandomVariable
            Stochastic process for the
            first differences
        differencing_order : int
            How many fold-differencing the
            the process represents. Must be
            an integer greater than or
            equal to 1. 1 represents a process
            on the first differences (the rate
            of change), 2 a process on the
            2nd differences (rate of change of
            the rate of change), et cetera.

        **kwargs :
            Additional keyword arguments passed to
            the parent class constructor.

        Returns
        -------
        None

        Notes
        -----
        The order of differencing is the discrete
        analogue of the order of a derivative in single
        variable calculus. A first difference (derivative)
        represents a rate of change. A second difference
        (derivative) represents the rate of change of that
        rate of change, et cetera.
        """
        self.assert_valid_differencing_order(differencing_order)
        self.differencing_order = differencing_order
        self.fundamental_process = fundamental_process
        super().__init__(name=name, **kwargs)

    def integrate(self, init_diffs: ArrayLike, highest_order_diffs: ArrayLike):
        """
        Integrate (de-difference) the differenced process,
        obtaining the process values X(0), X(1), ... X(t)
        from the n-th differences and a set of initial process /
        difference values X(0), X^1(0), X^2(0), ... X^(n-1)(0),
        where X^k(0) is the value of the n-th difference
        at the starting point of the process.

        Parameters
        ----------
        init_diffs : ArrayLike
            Values of X(0), X^1(0), X^2(0) ... X^(n-1)(0).

        highest_order_diffs : ArrayLike
            Array of differences at the highest of differencing,
            i.e. the order of the overall process.

        Returns
        -------
        The integrated (de-differenced) sequence of values.
        """
        if not init_diffs.size == self.differencing_order:
            raise ValueError(
                "Must have exactly as many "
                "initial difference values as "
                "the differencing order, given "
                "in the sequence X(0), X^1(0), etc"
                f"got {init_diffs.size} values "
                "for a process of order "
                f"{self.differencing_order}"
            )

        def _integrate_one_step(i, val):
            # numpydoc ignore=GL08
            return jnp.hstack([init_diffs[-i] + jnp.cumsum(val)])

        return jax.lax.fori_loop(
            0,
            self.differencing_order,
            _integrate_one_step,
            init_val=highest_order_diffs,
        )

    def sample(
        self,
        init_vals,
        **kwargs,
    ) -> tuple:
        """
        Sample from the process

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments passed to self.fundamental_process.sample()

        Returns
        -------
        SampledValue
            Whose value entry is a single array representing the
            undifferenced timeseries
        """
        diffs = self.fundamental_process.sample(**kwargs)
        return SampledValue(
            value=self.integrate(init_vals, diffs),
            t_start=self.t_start,
            t_unit=self.t_unit,
        )

    @staticmethod
    def validate():
        """
        Validates input parameters, implementation pending.
        """
        return None

    @staticmethod
    def assert_valid_differencing_order(differencing_order: any):
        """
        To be valid, a differencing order must
        be an integer and must be strictly positive.
        This function raises a value error if its
        argument is not a valid differencing order.

        Parameter
        ---------
        differcing_order : any
            Potential differencing order to validate.

        Returns
        -------
        None or raises a ValueError
        """
        if not isinstance(differencing_order, int):
            raise ValueError(
                "differcing_order must be an integer."
                f"got {type(differencing_order)}"
            )
        if not differencing_order >= 1:
            raise ValueError(
                "differcing_order must be an integer "
                "greater than or equal to 1. Got "
                f"{differencing_order}"
            )
