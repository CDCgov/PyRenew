# numpydoc ignore=GL08
"""
Linked ascertainment models.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
from jax.typing import ArrayLike

from pyrenew.ascertainment.base import AscertainmentModel
from pyrenew.latent import TemporalProcess
from pyrenew.metaclass import RandomVariable


class RatioLinkedAscertainment(AscertainmentModel):
    """
    Two ascertainment rates expressed as a base rate and a ratio.

    The linked ascertainment rate is the sampled base rate multiplied by the
    sampled ratio. Optional temporal processes act on the resulting scalar
    baselines and do not preserve the ratio pointwise over time.
    """

    def __init__(
        self,
        name: str,
        base_signal: str,
        linked_signal: str,
        base_rate_rv: RandomVariable,
        ratio_rv: RandomVariable,
        temporal_processes: Mapping[str, TemporalProcess] | None = None,
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
        temporal_processes
            Optional temporal processes keyed by signal name. Signals without
            a process retain their scalar baseline rate.
        """
        super().__init__(
            name=name,
            signals=(base_signal, linked_signal),
            temporal_processes=temporal_processes,
        )
        self.base_signal = base_signal
        self.linked_signal = linked_signal
        self.base_rate_rv = base_rate_rv
        self.ratio_rv = ratio_rv

    def _sample_baseline_rates(self) -> Mapping[str, ArrayLike]:
        """Sample the base rate and ratio and calculate the linked baseline.

        Returns
        -------
        Mapping[str, ArrayLike]
            Scalar baseline rates for the base and linked signals.
        """
        base_rate = jnp.asarray(self.base_rate_rv())
        ratio = jnp.asarray(self.ratio_rv())
        linked_rate = base_rate * ratio

        return {
            self.base_signal: base_rate,
            self.linked_signal: linked_rate,
        }
