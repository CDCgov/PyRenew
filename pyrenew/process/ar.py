# numpydoc ignore=GL08

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan
from numpyro.infer.reparam import LocScaleReparam

from pyrenew.metaclass import RandomVariable, SampledValue


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
    ) -> tuple:
        """
        Sample from the AR process

        Parameters
        ----------
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
        tuple
            With a single SampledValue containing an
            array of shape (n,) + init_vals.shape.
        """
        noise_sd_arr = jnp.atleast_1d(noise_sd)
        if not noise_sd_arr.shape == (1,):
            raise ValueError("noise_sd must be a scalar. " f"Got {noise_sd}")
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

            new_term = (
                jnp.tensordot(autoreg, recent_vals, axes=[0, 0]) + next_noise
            )
            new_recent_vals = jnp.vstack(
                [new_term, recent_vals[: (order - 1), ...]]
            )
            return new_recent_vals, new_term

        last, ts = scan(
            f=transition,
            init=init_vals[..., jnp.newaxis],
            xs=None,
            length=(n - order),
        )
        return (
            SampledValue(
                jnp.squeeze(
                    jnp.concatenate(
                        [init_vals[::, jnp.newaxis], ts],
                    )
                ),
                t_start=self.t_start,
                t_unit=self.t_unit,
            ),
        )

    @staticmethod
    def validate():  # numpydoc ignore=RT01
        """
        Validates input parameters, implementation pending.
        """
        return None
