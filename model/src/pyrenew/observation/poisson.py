# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable, TimeArray


class PoissonObservation(RandomVariable):
    """
    Poisson observation process
    """

    def __init__(
        self,
        parameter_name: str = "poisson_rv",
        eps: float = 1e-8,
    ) -> None:
        """
        Default Constructor

        Parameters
        ----------
        parameter_name : str, optional
            Passed to numpyro.sample. Defaults to "poisson_rv"
        eps : float, optional
            Small value added to the rate parameter to avoid zero values.
            Defaults to 1e-8.

        Returns
        -------
        None
        """

        self.parameter_name = parameter_name
        self.eps = eps

        return None

    def sample(
        self,
        mu: ArrayLike,
        obs: ArrayLike | None = None,
        name: str | None = None,
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
        name : str | None, optional
            Name of the random variable. Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        tuple
        """

        if name is None:
            name = self.parameter_name

        return (
            TimeArray(numpyro.sample(
                name=name,
                fn=dist.Poisson(rate=mu + self.eps),
                obs=obs,
            )),
        )

    @staticmethod
    def validate():  # numpydoc ignore=GL08
        None
