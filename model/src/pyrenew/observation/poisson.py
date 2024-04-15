# -*- coding: utf-8 -*-

import numpyro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class PoissonObservation(RandomVariable):
    """
    Poisson observation process

    Methods
    -------
    sample(predicted, obs, **kwargs)
        Sample from the Poisson process
    """

    def __init__(
        self,
        parameter_name: str | None = "poisson_rv",
        eps: float | None = 1e-8,
    ) -> None:
        """Default Constructor

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
        predicted: ArrayLike,
        obs: ArrayLike | None = None,
        **kwargs,
    ) -> tuple:
        """Sample from the Poisson process

        Parameters
        ----------
        predicted : ArrayLike
            Rate parameter of the Poisson distribution.
        obs : ArrayLike, optional
            Observed data. Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        tuple
        """
        return (
            numpyro.sample(
                self.parameter_name,
                dist.Poisson(rate=predicted + self.eps),
                obs=obs,
            ),
        )

    @staticmethod
    def validate():
        None
