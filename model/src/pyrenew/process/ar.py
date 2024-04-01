# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import lax
from numpy.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class ARProcess(RandomVariable):
    """
    Object to represent
    an AR(p) process in
    Numpyro
    """

    def __init__(
        self,
        mean: float,
        autoreg: ArrayLike,
        noise_sd: float,
    ) -> None:
        """Default constructor

        Parameters
        ----------
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
        self.mean = mean
        self.autoreg = autoreg
        self.noise_sd = noise_sd

    def sample(
        self,
        duration: int,
        inits: ArrayLike = None,
        name: str = "arprocess",
    ) -> tuple:
        """Sample from the AR process

        Parameters
        ----------
        duration: int
            Length of the sequence.
        inits : ArrayLike, optional
            Initial points, if None, then these are sampled.
        name : str, optional
            Name of the parameter passed to numpyro.sample.

        Returns
        -------
        tuple
        """
        order = self.autoreg.shape[0]
        if inits is None:
            inits = numpyro.sample(
                name + "_sampled_inits",
                dist.Normal(0, self.noise_sd).expand((order,)),
            )

        def _ar_scanner(carry, next):
            new_term = (jnp.dot(self.autoreg, carry) + next).flatten()
            new_carry = jnp.hstack([new_term, carry[: (order - 1)]])
            return new_carry, new_term

        noise = numpyro.sample(
            name + "_noise", dist.Normal(0, self.noise_sd).expand((duration,))
        )

        last, ts = lax.scan(_ar_scanner, inits - self.mean, noise)
        return (jnp.hstack([inits, self.mean + ts.flatten()]),)

    @staticmethod
    def validate():
        return None
