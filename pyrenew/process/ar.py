"""
This file defines a RandomVariable subclass for
autoregressive (AR) processes
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan
from numpyro.infer.reparam import LocScaleReparam

from pyrenew.metaclass import RandomVariable


class ARProcess(RandomVariable):
    """
    RandomVariable representing an
    an AR(p) process.
    """

    def sample(
        self,
        noise_name: str,
        n: int,
        autoreg: ArrayLike,
        init_vals: ArrayLike,
        noise_sd: float | ArrayLike,
    ) -> ArrayLike:
        """
        Sample from the AR process

        Parameters
        ----------
        noise_name
            A name for the sample site holding the
            Normal(`0`, `noise_sd`) noise for the AR process.
            Passed to [`numpyro.primitives.sample`][].
        n
            Length of the sequence.
        autoreg
            Autoregressive coefficients.
            The length of the array's first
            dimension determines the order $p$
            of the AR process.
        init_vals
            Array of initial values. Must have the
            same first dimension size as the order.
        noise_sd
            Standard deviation of the AR
            process Normal noise, which by
            definition has mean 0.

        Returns
        -------
        ArrayLike
            with first dimension of length `n`
            and additional dimensions as inferred
            from the shapes of `autoreg`,
            `init_vals`, and `noise_sd`.

        Notes
        -----
        The first dimension of the return value
        with be of length `n` and represents time.
        Trailing dimensions follow standard numpy
        broadcasting rules and are determined from
        the second through `n` th dimensions, if any,
        of `autoreg` and `init_vals`, as well as the
        all dimensions of `noise_sd` (i.e.
        `jax.numpy.shape(autoreg)[1:]`,
        `jax.numpy.shape(init_vals)[1:]`
        and `jax.numpy.shape(noise_sd)`

        Those shapes must be
        broadcastable together via
        [`jax.lax.broadcast_shapes`][]. This can
        be used to produce multiple AR processes of the
        same order but with either shared or different initial
        values, AR coefficient vectors, and/or
        and noise standard deviation values.
        """
        autoreg = jnp.atleast_1d(autoreg)
        init_vals = jnp.atleast_1d(init_vals)
        noise_sd = jnp.array(noise_sd)
        # noise_sd can be a scalar, but
        # autoreg and init_vals must have a
        # a first dimension (time),
        # as the order of the process is
        # inferred from that first dimension

        order = autoreg.shape[0]
        n_inits = init_vals.shape[0]

        try:
            noise_shape = jax.lax.broadcast_shapes(
                init_vals.shape[1:],
                autoreg.shape[1:],
                noise_sd.shape,
            )
        except Exception as e:
            raise ValueError(
                "Could not determine a "
                "valid shape for the AR process noise "
                "from the shapes of the init_vals, "
                "autoreg, and noise_sd arrays. "
                "See ARProcess.sample() documentation "
                "for details."
            ) from e

        if not n_inits == order:
            raise ValueError(
                "Initial values array must have the same "
                "first dimension length as the order p of "
                "the AR process. The order is given by "
                "the first dimension length of the array "
                "of autoregressive coefficients. Got an initial "
                f"value array with first dimension {n_inits} for "
                f"a process of order {order}"
            )

        history_shape = (order,) + noise_shape

        try:
            inits_broadcast = jnp.broadcast_to(init_vals, history_shape)
        except Exception as e:
            raise ValueError(
                "Could not broadcast init_vals "
                f"(shape {init_vals.shape}) "
                "to the expected shape of the process "
                f"history (shape {history_shape}). "
                "History shape is determined by the "
                "shapes of the init_vals, autoreg, and "
                "noise_sd arrays. See ARProcess "
                "documentation for details"
            ) from e

        inits_flipped = jnp.flip(inits_broadcast, axis=0)

        def transition(recent_vals, _):  # numpydoc ignore=GL08
            with numpyro.handlers.reparam(config={noise_name: LocScaleReparam(0)}):
                next_noise = numpyro.sample(
                    noise_name,
                    numpyro.distributions.Normal(
                        loc=jnp.zeros(noise_shape), scale=noise_sd
                    ),
                )

            dot_prod = jnp.einsum("i...,i...->...", autoreg, recent_vals)
            new_term = dot_prod + next_noise
            new_recent_vals = jnp.concatenate(
                [
                    new_term[jnp.newaxis, ...],
                    # concatenate as (1 time unit,) + noise_shape
                    # array
                    recent_vals,
                ],
                axis=0,
            )[:order]

            return new_recent_vals, new_term

        if n > order:
            _, ts = scan(
                f=transition,
                init=inits_flipped,
                xs=None,
                length=(n - order),
            )

            ts_with_inits = jnp.concatenate(
                [inits_broadcast, ts],
                axis=0,
            )
        else:
            ts_with_inits = inits_broadcast
        return ts_with_inits[:n]

    @staticmethod
    def validate():  # numpydoc ignore=RT01
        """
        Validates input parameters, implementation pending.
        """
        return None
