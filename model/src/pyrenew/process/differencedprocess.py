# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan
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

    def integrate(
        self, init_diff_vals: ArrayLike, highest_order_diff_vals: ArrayLike
    ):
        """
        Integrate (de-difference) the differenced process,
        obtaining the process values X(t=0), X(t=1), ... X(t)
        from the n-th differences and a set of initial process /
        difference values X(t=0), X^1(t=1), X^2(t=2), ...
        X^(n-1)(t=n-1), where X^k(t) is the value of the n-th
        differenc at index t of the process.

        Parameters
        ----------
        init_diff_vals : ArrayLike
            Values of X(t=0), X^1(t=1), X^2(t=2) ... X^(n-1)(t=n-1).

        highest_order_diff_vals : ArrayLike
            Array of differences at the highest order of
            differencing, i.e. the order of the overall process,
            starting with X^n(t=n)

        Returns
        -------
        The integrated (de-differenced) sequence of values.
        """
        if not init_diff_vals.size == self.differencing_order:
            raise ValueError(
                "Must have exactly as many "
                "initial difference values as "
                "the differencing order, given "
                "in the sequence X(t=0), X^1(t=1), etc"
                f"got {init_diff_vals.size} values "
                "for a process of order "
                f"{self.differencing_order}"
            )

        def _integrate_one_step(diffs, scanned):
            # numpydoc ignore=GL08
            order, init = scanned
            new_diffs = jnp.cumsum(diffs.at[order].set(init))
            return (new_diffs, None)

        integrated, _ = scan(
            _integrate_one_step,
            init=jnp.pad(
                highest_order_diff_vals, (self.differencing_order, 0)
            ),
            xs=(
                jnp.flip(jnp.arange(self.differencing_order)),
                jnp.flip(init_diff_vals),
            ),
        )

        return integrated

    def sample(
        self,
        init_vals: ArrayLike,
        *args,
        **kwargs,
    ) -> tuple:
        """
        Sample from the process

        Parameters
        ----------
        init_vals : ArrayLike
            initial values for the differenced process,
            passed as the init_diff_vals to DifferencedProcess.integrate

        *args :
           Additional positional arguments passed to
           self.fundamental_process.sample()

        **kwargs : dict, optional
            Keyword arguments passed to self.fundamental_process.sample()

        Returns
        -------
        SampledValue
            Whose value entry is a single array representing the
            undifferenced timeseries
        """
        diffs, *_ = self.fundamental_process.sample(*args, **kwargs)
        return (
            SampledValue(
                value=self.integrate(init_vals, diffs.value),
                t_start=self.t_start,
                t_unit=self.t_unit,
            ),
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
