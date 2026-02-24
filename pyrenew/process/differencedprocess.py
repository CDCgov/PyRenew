# numpydoc ignore=GL08

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax.typing import ArrayLike

from pyrenew.math import integrate_discrete
from pyrenew.metaclass import RandomVariable


class DifferencedProcess(RandomVariable):
    """
    Class for differenced stochastic process $X(t)$,
    constructed by placing a fundamental stochastic
    process on the $n^{th}$ differences
    (rates of change). See
    https://otexts.com/fpp3/stationarity.html
    for a discussion of differencing in the
    context of discrete timeseries data.

    Notes
    -----
    The order of differencing is the discrete
    analogue of the order of a derivative in single
    variable calculus. A first difference (derivative)
    represents a rate of change. A second difference
    (derivative) represents the rate of change of that
    rate of change, et cetera.
    """

    def __init__(
        self,
        name: str,
        fundamental_process: RandomVariable,
        differencing_order: int,
        **kwargs: object,
    ) -> None:
        """
        Default constructor.

        Parameters
        ----------
        name
            A name for this random variable.
        fundamental_process
            Stochastic process for the
            differences. Must accept an
            `n` argument specifying the number
            of samples to draw.
        differencing_order
            How many fold-differencing the
            the process represents. Must be
            an integer greater than or
            equal to 1. 1 represents a process
            on the first differences (the rate
            of change), 2 a process on the
            2nd differences (rate of change of
            the rate of change), et cetera.

        Returns
        -------
        None
        """
        self.assert_valid_differencing_order(differencing_order)
        self.fundamental_process = fundamental_process
        self.differencing_order = differencing_order
        super().__init__(name=name, **kwargs)

    @staticmethod
    def assert_valid_differencing_order(differencing_order: Any) -> None:  # noqa: ANN401
        """
        To be valid, a differencing order must
        be an integer and must be strictly positive.
        This function raises a value error if its
        argument is not a valid differencing order.

        Parameters
        ----------
        differencing_order
            Potential differencing order to validate.

        Returns
        -------
        None
            or raises a [`ValueError`][]
        """
        if not isinstance(differencing_order, int):
            raise ValueError(
                "differencing_order must be an integer. "
                f"got type {type(differencing_order)} "
                f"and value {differencing_order}"
            )
        if not differencing_order >= 1:
            raise ValueError(
                "differencing_order must be an integer "
                "greater than or equal to 1. Got "
                f"{differencing_order}"
            )

    def validate(self) -> None:
        """
        Empty validation method.
        """
        pass

    def sample(
        self,
        init_vals: ArrayLike,
        n: int,
        *args: object,
        fundamental_process_init_vals: ArrayLike = None,
        **kwargs: object,
    ) -> jnp.ndarray:
        """
        Sample from the process

        Parameters
        ----------
        init_vals
            initial values for the $0^{th}$ through
            $(n-1)^{st}$ differences, passed as the
            `init_diff_vals` argument to
            [`pyrenew.math.integrate_discrete`][].

        n
            Number of values to sample. Will sample
            `n - differencing_order` values from
            `self.fundamental_process` to ensure
            that the de-differenced output is of length
            `n`.

        *args
            Additional positional arguments passed to
            `self.fundamental_process.sample`

        fundamental_process_init_vals
            Initial values for the fundamental process.
            Passed as the `init_vals` keyword argument
            to `self.fundamental_process.sample`.
            Default `None`.

        **kwargs
            Keyword arguments passed to
            `self.fundamental_process.sample()`.

        Returns
        -------
        jnp.ndarray
            An array representing the undifferenced timeseries
        """
        if not isinstance(n, int):
            raise ValueError(f"n must be an integer. Got {type(n)}")
        if n < 1:
            raise ValueError(f"n must be positive. Got {n}")

        init_vals = jnp.atleast_1d(init_vals)
        n_inits = init_vals.shape[0]

        if not n_inits == self.differencing_order:
            raise ValueError(
                "Must have exactly as many "
                "initial difference values as "
                "the differencing order, given "
                "in the sequence $X(t=0), X^1(t=1),$ "
                "et cetera. "
                f"Got {n_inits} values "
                "for a process of order "
                f"{self.differencing_order}."
            )
        n_diffs = n - self.differencing_order

        if n_diffs > 0:
            diff_samp = self.fundamental_process.sample(
                *args,
                n=n_diffs,
                init_vals=fundamental_process_init_vals,
                **kwargs,
            )
            diffs = diff_samp
        else:
            diffs = jnp.array([])
        integrated_ts = integrate_discrete(init_vals, diffs)[:n]
        return integrated_ts
