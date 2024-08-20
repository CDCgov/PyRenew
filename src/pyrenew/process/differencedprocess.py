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
    process on the :math:`n^{th}` differences
    (rates of change). See
    https://otexts.com/fpp2/stationarity.html
    for a discussion of differencing in the
    context of discrete timeseries data.
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
            first differences. Should accept an
            `n` argument specifying the number
            of samples to draw.
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
        obtaining the process values :math:`X(t=0), X(t=1), ... X(t)`
        from the :math:`n^{th}` differences and a set of
        initial process / difference values
        :math:`X(t=0), X^1(t=1), X^2(t=2), ... X^{(n-1)}(t=n-1)`,
        where :math:`X^k(t)` is the value of the :math:`n^{th}`
        difference at index :math:`t` of the process.

        Parameters
        ----------
        init_diff_vals : ArrayLike
            Values of
            :math:`X(t=0), X^1(t=1), X^2(t=2) ... X^(n-1)(t=n-1)`.

        highest_order_diff_vals : ArrayLike
            Array of differences at the highest order of
            differencing, i.e. the order of the overall process,
            starting with :math:`X^{(n-1)}(t=n-1)`

        Returns
        -------
        The integrated (de-differenced) sequence of values.
        """
        init_diff_vals = jnp.atleast_1d(init_diff_vals)
        highest_order_diff_vals = jnp.atleast_1d(highest_order_diff_vals)
        n_inits = init_diff_vals.shape[0]
        if not n_inits == self.differencing_order:
            raise ValueError(
                "Must have exactly as many "
                "initial difference values as "
                "the differencing order, given "
                "in the sequence :math:`X(t=0), X^1(t=1),` "
                "et cetera. "
                f"Got {n_inits} values "
                "for a process of order "
                f"{self.differencing_order}"
            )
        init_row_shape = init_diff_vals[0].shape
        diff_row_shape = highest_order_diff_vals[0].shape
        if not init_row_shape == diff_row_shape:
            raise ValueError(
                "First axis entries for init_diff_vals "
                "and highest_order_diff_vals (e.g. "
                "init_diff_vals[0], "
                "highest_order_diff_vals[1], et cetera) "
                "must either both be scalars (shape ()) "
                "or both be arrays of identical shape. "
                f"Got shape {init_row_shape} for the "
                "init_diff_vals "
                f"and shape {diff_row_shape} for the "
                "highest_order_diff_vals"
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
        n: int,
        *args,
        fundamental_process_init_vals: ArrayLike = None,
        **kwargs,
    ) -> tuple:
        """
        Sample from the process

        Parameters
        ----------
        init_vals : ArrayLike
            initial values for the :math:`0^{th}` through
            math:`(n-1)^{st}` differences, passed as the
            :param:`init_diff_vals` argument to
            :meth:`DifferencedProcess.integrate()`

        n : int
            Number of values to sample. Will sample n - 1
            values from :meth:`self.fundamental_process`.

        *args :
           Additional positional arguments passed to
           :meth:`self.fundamental_process.sample()`

        fundamental_process_init_vals : ArrayLike
           Initial values for the fundamental process.
           Passed as the :param:`init_vals` keyword argument
           to :meth:`self.fundamental_process.sample()`.

        **kwargs : dict, optional
            Keyword arguments passed to
            :meth:`self.fundamental_process.sample()`.

        Returns
        -------
        SampledValue
            Whose value entry is a single array representing the
            undifferenced timeseries
        """
        diffs, *_ = self.fundamental_process.sample(
            *args, n=(n - 1), init_vals=fundamental_process_init_vals, **kwargs
        )
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
