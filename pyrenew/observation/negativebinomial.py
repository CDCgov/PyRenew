# numpydoc ignore=GL08

from __future__ import annotations

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
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name
            Name for the numpyro variable.
        concentration_rv
            Random variable from which to sample the positive concentration
            parameter of the negative binomial. This parameter is sometimes
            called k, phi, or the "dispersion" or "overdispersion" parameter,
            despite the fact that larger values imply that the distribution
            becomes more Poissonian, while smaller ones imply a greater degree
            of dispersion.

        Returns
        -------
        None
        """
        super().__init__(name=name)
        self.concentration_rv = concentration_rv

    def sample(
        self,
        mu: ArrayLike,
        obs: ArrayLike | None = None,
        **kwargs: object,
    ) -> ArrayLike:
        """
        Sample from the negative binomial distribution

        Parameters
        ----------
        mu
            Mean parameter of the negative binomial distribution.
        obs
            Observed data, by default None.
        **kwargs
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        ArrayLike
        """
        concentration = self.concentration_rv.sample()

        negative_binomial_sample = numpyro.sample(
            name=self.name,
            fn=dist.NegativeBinomial2(
                mean=mu,
                concentration=concentration,
            ),
            obs=obs,
        )

        return negative_binomial_sample
