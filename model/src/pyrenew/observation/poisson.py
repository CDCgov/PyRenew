# -*- coding: utf-8 -*-

import numpyro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class PoissonObservation(RandomVariable):
    """
    Poisson observation process
    """

    def __init__(
        self,
        parameter_name: str = "poisson_rv",
        eps: float = 1e-8,
    ) -> None:
        """Default Constructor

        Parameters
        ----------
        parameter_name : str, optional
            Passed to numpyro.sample.
        eps : float, optional
            Small value added to the rate parameter to avoid zero values.

        Returns
        -------
        None
        """

        self.parameter_name = parameter_name
        self.eps = eps

        return None

    def sample(
        self,
        mean: ArrayLike,
        obs: ArrayLike = None,
        **kwargs,
    ) -> tuple:
        """Sample from the Poisson process

        Parameters
        ----------
        mean : ArrayLike
            Rate parameter of the Poisson distribution.
        obs : ArrayLike, optional
            Observed data, by default None.
        kwargs : dict
            Keyword arguments passed to the sampling methods.

        Returns
        -------
        tuple
        """
        return (
            numpyro.sample(
                self.parameter_name,
                dist.Poisson(rate=mean + self.eps),
                obs=obs,
            ),
        )

    @staticmethod
    def validate():
        None
