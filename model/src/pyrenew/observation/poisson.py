# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable, SampledValue


class PoissonObservation(RandomVariable):
    """
    Poisson observation process
    """

    def __init__(
        self,
        name: str,
        eps: float = 1e-8,
    ) -> None:
        """
        Default Constructor

        Parameters
        ----------
        name : str, optional
            Passed to numpyro.sample.
        eps : float, optional
            Small value added to the rate parameter to avoid zero values.
            Defaults to 1e-8.

        Returns
        -------
        None
        """

        self.name = name
        self.eps = eps

        return None

    @staticmethod
    def validate():  # numpydoc ignore=GL08
        None

    def sample(
        self,
        mu: ArrayLike,
        obs: ArrayLike | None = None,
        **kwargs,
    ) -> tuple:
        """
        Sample from the Poisson process

        Parameters
        ----------
        mu : ArrayLike
            Rate parameter of the Poisson distribution.
        obs : ArrayLike | None, optional
            Observed data. Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        tuple
        """

        poisson_sample = numpyro.sample(
            name=self.name,
            fn=dist.Poisson(rate=mu + self.eps),
            obs=obs,
        )
        return (SampledValue(poisson_sample),)
