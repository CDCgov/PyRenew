# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable, SampledValue


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
            Name for the numpyro variable.
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
        self.concentration_rv = concentration_rv
        self.eps = eps

    @staticmethod
    def validate(concentration_rv: RandomVariable) -> None:
        """
        Check that the concentration_rv is actually a RandomVariable

        Parameters
        ----------
        concentration_rv : any
            RandomVariable from which to sample the positive concentration
            parameter of the negative binomial.

        Returns
        -------
        None
        """
        assert isinstance(concentration_rv, RandomVariable)
        return None

    def sample(
        self,
        mu: ArrayLike,
        obs: ArrayLike | None = None,
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
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        tuple
        """
        concentration, *_ = self.concentration_rv.sample()

        negative_binomial_sample = numpyro.sample(
            name=self.name,
            fn=dist.NegativeBinomial2(
                mean=mu + self.eps,
                concentration=concentration.array,
            ),
            obs=obs,
        )

        return (SampledValue(negative_binomial_sample),)
