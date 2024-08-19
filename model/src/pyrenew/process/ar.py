# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan
from pyrenew.metaclass import RandomVariable, SampledValue
from pyrenew.process.iidrandomsequence import StandardNormalSequence


class ARProcess(RandomVariable):
    """
    RandomVariable representing an
    an AR(p) process.
    """

    def __init__(self, name, *args, noise_rv_suffix: str = "_noise", **kwargs):
        """
        Default constructor.

        Parameters
        ----------
        name : str
           A name for the process.

        noise_rv_suffix : str
           A suffix to append to name when naming the
           internal RandomVariable holding the process
           noise. Default `"_noise"`.
        """
        self.name = name
        super().__init__(*args, **kwargs)
        self.noise_rv_ = StandardNormalSequence(name=name + noise_rv_suffix)

    def sample(
        self,
        n: int,
        autoreg: ArrayLike,
        init_vals: ArrayLike,
        noise_sd: ArrayLike,
        **kwargs,
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
        **kwargs : dict, optional
            Additional keyword arguments passed to
            self.noise_rv_.sample()

        Returns
        -------
        tuple
            With a single SampledValue containing an
            array of shape (n,).
        """
        order = autoreg.shape[0]
        raw_noise, *_ = self.noise_rv_(n=n, **kwargs)
        noise = noise_sd * raw_noise.value

        def transition(recent_vals, next_noise):  # numpydoc ignore=GL08
            new_term = jnp.dot(autoreg, recent_vals) + next_noise
            new_recent_vals = jnp.hstack(
                [new_term, recent_vals[: (order - 1)]]
            )
            return new_recent_vals, new_term

        last, ts = scan(transition, init_vals, noise)
        return (
            SampledValue(
                jnp.hstack([init_vals, ts]),
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
