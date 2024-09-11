# numpydoc ignore=GL08

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan
from numpyro.infer.reparam import LocScaleReparam


from pyrenew.metaclass import RandomVariable
from pyrenew.process.iidrandomsequence import StandardNormalSequence


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
        noise_sd: float | ArrayLike
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

        noise_shape = jax.lax.broadcast_shapes(
            autoreg.shape[1:], noise_sd.shape
        )

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

            print("dot product shape: ", dot_prod.shape)
            print("next noise shape: ", next_noise.shape)

            new_term = jnp.atleast_1d(dot_prod + next_noise)

            recent_to_concat = recent_vals[: (order - 1), ...]

            print("new term shape: ", new_term.shape)
            print("recent to concat shape: ", recent_to_concat.shape)
            new_recent_vals = jnp.vstack([new_term, recent_to_concat])

            return new_recent_vals, new_term

        inits_flipped = jnp.flip(init_vals, axis=0)

        last, ts = scan(
            f=transition,
            init=inits_flipped,
            xs=None,
            length=(n - order),
        )

        result = jnp.atleast_1d(
            jnp.squeeze(
                jnp.vstack(
                    [
                        init_vals[..., jnp.newaxis].reshape(
                            (order,) + ts.shape[1:]
                        ),
                        ts,
                    ],
                )[:n, ...]
            )
        )

        return result

    @staticmethod
    def validate():  # numpydoc ignore=RT01
        """
        Validates input parameters, implementation pending.
        """
        return None
