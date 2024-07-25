# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import lax
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable, SampledValue


class ARProcess(RandomVariable):
    """
    Object to represent
    an AR(p) process in
    Numpyro
    """

    def __init__(
        self,
        name: str,
        mean: float,
        autoreg: ArrayLike,
        noise_sd: float,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name : str
            Name of the parameter passed to numpyro.sample.
        mean: float
            Mean parameter.
        autoreg : ArrayLike
            Model parameters. The shape determines the order.
        noise_sd : float
            Standard error for the noise component.

        Returns
        -------
        None
        """
        self.name = name
        self.mean = mean
        self.autoreg = autoreg
        self.noise_sd = noise_sd

    def sample(
        self,
        duration: int,
        inits: ArrayLike = None,
        **kwargs,
    ) -> tuple:
        """
        Sample from the AR process

        Parameters
        ----------
        duration: int
            Length of the sequence.
        inits : ArrayLike, optional
            Initial points, if None, then these are sampled.
            Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        tuple
            With a single array of shape (duration,).
        """
        order = self.autoreg.shape[0]
        if inits is None:
            inits = numpyro.sample(
                self.name + "_sampled_inits",
                dist.Normal(0, self.noise_sd).expand((order,)),
            )

        def _ar_scanner(carry, next):  # numpydoc ignore=GL08
            new_term = (jnp.dot(self.autoreg, carry) + next).flatten()
            new_carry = jnp.hstack([new_term, carry[: (order - 1)]])
            return new_carry, new_term

        noise = numpyro.sample(
            self.name + "_noise",
            dist.Normal(0, self.noise_sd).expand((duration - inits.size,)),
        )

        last, ts = lax.scan(_ar_scanner, inits - self.mean, noise)
        return (SampledValue(jnp.hstack([inits, self.mean + ts.flatten()])),)

    @staticmethod
    def validate():  # numpydoc ignore=RT01
        """
        Validates inputted parameters, implementation pending.
        """
        return None
