# numpydoc ignore=GL08

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
        noise_name: str
            A name for the sample site holding the
            Normal(0, noise_sd) noise for the AR process.
            Passed to :func:`numpyro.sample`.
        n: int
            Length of the sequence.
        autoreg: ArrayLike
            Autoregressive coefficients.
            The length of the array's first
            dimension determines the order :math`p`
            of the AR process.
        init_vals : ArrayLike
            Array of initial values. Must have the
            same first dimension size as the order.
        noise_sd : ArrayLike
            Standard deviation of the AR
            process Normal noise, which by
            definition has mean 0.

        Returns
        -------
        ArrayLike
            of shape (n,) + init_vals.shape[1:].
        """
        autoreg = jnp.atleast_1d(autoreg)
        noise_sd = jnp.atleast_1d(noise_sd)
        init_vals = jnp.atleast_1d(init_vals)
        order = autoreg.shape[0]

        if not init_vals.shape == autoreg.shape:
            raise ValueError(
                "Initial values array and autoregressive "
                "coefficient array must be of the same shape ",
                "and must have a first dimension that represents "
                "the order of the AR process. Got a shape of "
                "{init_vals.shape} for the initial values and "
                "a shape of {autoreg.shape} for the autoregressive "
                "coefficients",
            )

        noise_shape = jax.lax.broadcast_shapes(
            autoreg.shape[1:], noise_sd.shape
        )

        term_shape = (1,) + noise_shape
        history_shape = (order,) + noise_shape
        subset_shape = (order - 1,) + noise_shape

        def transition(recent_vals, _):  # numpydoc ignore=GL08
            with numpyro.handlers.reparam(
                config={noise_name: LocScaleReparam(0)}
            ):
                next_noise = numpyro.sample(
                    noise_name,
                    numpyro.distributions.Normal(
                        loc=jnp.zeros(noise_shape), scale=noise_sd
                    ),
                )

            dot_prod = jnp.einsum("i...,i...->...", autoreg, recent_vals)

            new_term = (dot_prod + next_noise)[jnp.newaxis, ...]

            assert jnp.shape(new_term) == term_shape

            if order > 1:
                recent_to_concat = recent_vals[: (order - 1)]
                assert jnp.shape(recent_to_concat) == subset_shape
                new_recent_vals = jnp.concatenate(
                    [new_term, recent_to_concat], axis=0
                )
            else:
                new_recent_vals = new_term

            return new_recent_vals, new_term

        inits_flipped = jnp.broadcast_to(
            jnp.flip(init_vals, axis=0), history_shape
        )

        assert jnp.shape(inits_flipped) == history_shape

        if n - order > 0:
            last, ts = scan(
                f=transition,
                init=inits_flipped,
                xs=None,
                length=(n - order),
            )

            ts_with_inits = jnp.concatenate(
                [
                    jnp.broadcast_to(init_vals, history_shape),
                    ts.reshape((n - order, -1)),
                ],
                axis=0,
            )
        else:
            ts_with_inits = init_vals

        return jnp.atleast_1d(jnp.squeeze(ts_with_inits[:n]))

    @staticmethod
    def validate():  # numpydoc ignore=RT01
        """
        Validates input parameters, implementation pending.
        """
        return None
