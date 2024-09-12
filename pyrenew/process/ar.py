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
        noise_sd = jnp.array(noise_sd)
        order = autoreg.shape[0]
        noise_shape = jax.lax.broadcast_shapes(
            autoreg.shape[1:], noise_sd.shape
        )
        n_inits = jnp.atleast_1d(init_vals).shape[0]

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
        term_shape = (1,) + noise_shape
        history_shape = (order,) + noise_shape

        inits_broadcast = jnp.broadcast_to(init_vals, history_shape)

        inits_flipped = jnp.flip(inits_broadcast, axis=0)
        assert jnp.shape(inits_flipped) == history_shape

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
                assert jnp.shape(next_noise) == noise_shape

            dot_prod = jnp.einsum("i...,i...->...", autoreg, recent_vals)

            new_term = (dot_prod + next_noise)[jnp.newaxis]

            assert jnp.shape(new_term) == term_shape

            new_recent_vals = jnp.concatenate([new_term, recent_vals], axis=0)[
                :order
            ]

            assert jnp.shape(new_recent_vals) == history_shape
            return new_recent_vals, new_term

        last, ts = scan(
            f=transition,
            init=inits_flipped,
            xs=None,
            length=(n - order),
        )

        ts_with_inits = jnp.concatenate(
            [inits_broadcast[::, jnp.newaxis, ...], ts],
            axis=0,
        )

        return jnp.atleast_1d(jnp.squeeze(ts_with_inits[:n]))

    @staticmethod
    def validate():  # numpydoc ignore=RT01
        """
        Validates input parameters, implementation pending.
        """
        return None
