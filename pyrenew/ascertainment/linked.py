# numpydoc ignore=GL08
"""
Linked ascertainment models.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpyro
from jax.typing import ArrayLike

from pyrenew.ascertainment.base import AscertainmentModel
from pyrenew.metaclass import RandomVariable


class RatioLinkedAscertainment(AscertainmentModel):
    """
    Two ascertainment rates expressed as a base rate and a ratio.

    The linked ascertainment rate is the sampled base rate multiplied by the
    sampled ratio.
    """

    def __init__(
        self,
        name: str,
        base_signal: str,
        linked_signal: str,
        base_rate_rv: RandomVariable,
        ratio_rv: RandomVariable,
    ) -> None:
        """
        Initialize a ratio-linked ascertainment model.

        Parameters
        ----------
        name
            Name of the ascertainment model.
        base_signal
            Name of the signal whose ascertainment rate is sampled directly.
        linked_signal
            Name of the signal whose ascertainment rate is the product of the
            base rate and ratio.
        base_rate_rv
            Random variable for the base signal's ascertainment rate.
        ratio_rv
            Random variable for the ratio of the linked signal's ascertainment
            rate to the base signal's ascertainment rate.
        """
        super().__init__(name=name, signals=(base_signal, linked_signal))
        self.base_signal = base_signal
        self.linked_signal = linked_signal
        self.base_rate_rv = base_rate_rv
        self.ratio_rv = ratio_rv

    def sample(self, **kwargs: object) -> Mapping[str, ArrayLike]:
        """
        Sample the base rate and ratio and calculate the linked rate.

        Parameters
        ----------
        **kwargs
            Additional model-context arguments, ignored.

        Returns
        -------
        Mapping[str, ArrayLike]
            Mapping from the base and linked signal names to their sampled
            ascertainment rates.
        """
        base_rate = self.base_rate_rv()
        ratio = self.ratio_rv()
        linked_rate = base_rate * ratio

        numpyro.deterministic(f"{self.name}_{self.base_signal}", base_rate)
        numpyro.deterministic(f"{self.name}_{self.linked_signal}", linked_rate)

        return {
            self.base_signal: base_rate,
            self.linked_signal: linked_rate,
        }
