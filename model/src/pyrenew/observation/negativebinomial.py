# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import numbers as nums

import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class NegativeBinomialObservation(RandomVariable):
    """Negative Binomial observation"""

    def __init__(
        self,
        name: str,
        concentration_rv: RandomVariable,
        eps: float = 1e-10,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name : str
            Name for the numpy variable.
        concentration : RandomVariable
            Random variable from which to sample the positive concentration
            parameter of the negative binomial. This parameter is sometimes
            called k, phi, or the "dispersion" or "overdispersion" parameter,
            despite the fact that larger values imply that the distribution
            becomes more Poissonian, while smaller ones imply a greater degree
            of dispersion.
        eps : float, optional
            Small value to add to the predicted mean to prevent numerical
            instability. Defaults to 1e-10.

        Returns
        -------
        None
        """

        NegativeBinomialObservation.validate(concentration_rv)

        self.name = name
        self.eps = eps

    def sample(
        self,
        mu: ArrayLike,
        obs: ArrayLike | None = None,
        name: str | None = None,
        **kwargs,
    ) -> tuple:
        """
        Sample from the negative binomial distribution

        Parameters
        ----------
        mu : ArrayLike
            Mean parameter of the negative binomial distribution.
        obs : ArrayLike, optional
            Observed data, by default None.
        name : str, optional
            Name of the random variable if other than that defined during
            construction, by default None (self.parameter_name).
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        tuple
        """
        concentration = self.concentration_rv()

        if name is None:
            name = self.parameter_name

        return (
            numpyro.sample(
                name=name,
                fn=dist.NegativeBinomial2(
                    mean=mu + self.eps,
                    concentration=concentration,
                ),
                obs=obs,
            ),
        )

    @staticmethod
    def validate(concentration_prior: any) -> None:
        """
        Check that the concentration prior is actually a nums.Number

        Parameters
        ----------
        concentration_prior : any
            Numpyro distribution from which to sample the positive concentration
            parameter of the negative binomial. Expected dist.Distribution or
            numbers.nums

        Returns
        -------
        None
        """
        assert isinstance(
            concentration_prior, (dist.Distribution, nums.Number)
        )
        return None
