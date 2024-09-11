# numpydoc ignore=GL08

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan

from pyrenew.metaclass import RandomVariable
from pyrenew.process.iidrandomsequence import StandardNormalSequence


class ARProcess(RandomVariable):
    """
    RandomVariable representing an
    an AR(p) process.
    """

    def __init__(self, noise_rv_name: str, *args, **kwargs) -> None:
        """
        Default constructor.

        Parameters
        ----------
        noise_rv_name : str
           A name for the internal RandomVariable
           holding the process noise.
        """
        super().__init__(*args, **kwargs)
        self.noise_rv_ = StandardNormalSequence(element_rv_name=noise_rv_name)

    def sample(
        self,
        n: int,
        autoreg: ArrayLike,
        init_vals: ArrayLike,
        noise_sd: float | ArrayLike,
        **kwargs,
    ) -> ArrayLike:
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
        noise_sd : float | ArrayLike
            Scalar giving the s.d. of the AR
            process Normal noise, which by
            definition has mean 0.
        **kwargs : dict, optional
            Additional keyword arguments passed to
            self.noise_rv_.sample()

        Returns
        -------
        ArrayLike
        """
        noise_sd_arr = jnp.atleast_1d(noise_sd)
        if not noise_sd_arr.shape == (1,):
            raise ValueError("noise_sd must be a scalar. " f"Got {noise_sd}")
        autoreg = jnp.atleast_1d(autoreg)
        init_vals = jnp.atleast_1d(init_vals)

        if not autoreg.ndim == 1:
            raise ValueError(
                "Array of autoregressive coefficients "
                "must be no more than 1 dimension",
                f"Got {autoreg.ndim}",
            )
        if not init_vals.ndim == 1:
            raise ValueError(
                "Array of initial values must be " "no more than 1 dimension",
                f"Got {init_vals.ndim}",
            )
        order = autoreg.size
        if not init_vals.size == order:
            raise ValueError(
                "Array of initial values must be "
                "be the same size as the order of "
                "the autoregressive process, "
                "which is determined by the number "
                "of autoregressive coefficients "
                "provided. Got {init_vals.size} "
                "initial values for a process of "
                f"order {order}"
            )

        raw_noise = self.noise_rv_(n=n, **kwargs)
        noise = noise_sd_arr * raw_noise

        def transition(recent_vals, next_noise):  # numpydoc ignore=GL08
            new_term = jnp.dot(autoreg, recent_vals) + next_noise
            new_recent_vals = jnp.hstack(
                [new_term, recent_vals[: (order - 1)]]
            )
            return new_recent_vals, new_term

        last, ts = scan(transition, init_vals, noise)
        return jnp.hstack([init_vals, ts])

    @staticmethod
    def validate():  # numpydoc ignore=RT01
        """
        Validates input parameters, implementation pending.
        """
        return None
