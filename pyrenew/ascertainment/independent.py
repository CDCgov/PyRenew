# numpydoc ignore=GL08
"""
Independent ascertainment models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from jax.typing import ArrayLike

from pyrenew.ascertainment.base import AscertainmentModel
from pyrenew.latent import TemporalProcess
from pyrenew.metaclass import RandomVariable


class IndependentAscertainment(AscertainmentModel):
    """Ascertainment rates with independently sampled scalar baselines.

    Use this model for one signal or for multiple signals whose baseline rates
    do not share a joint prior. Add temporal processes only for signals whose
    rates should vary over time.
    """

    def __init__(
        self,
        name: str,
        rate_rvs: Mapping[str, RandomVariable],
        temporal_processes: Mapping[str, TemporalProcess] | None = None,
    ) -> None:
        """Initialize independent ascertainment rates.

        Parameters
        ----------
        name
            Name of the ascertainment model.
        rate_rvs
            Non-empty mapping from signal names to random variables for their
            scalar baseline rates. Mapping order determines signal order.
        temporal_processes
            Optional temporal processes keyed by signal name. Signals without
            a process retain their scalar baseline rate.

        Raises
        ------
        TypeError
            If ``rate_rvs`` is not a mapping or a value is not a
            ``RandomVariable``.
        ValueError
            If ``rate_rvs`` is empty or contains an invalid signal name.
        """
        if not isinstance(rate_rvs, Mapping):
            raise TypeError(
                f"rate_rvs must be a mapping, got {type(rate_rvs).__name__}."
            )
        if len(rate_rvs) == 0:
            raise ValueError("rate_rvs must be a non-empty mapping.")

        for signal, rate_rv in rate_rvs.items():
            if not isinstance(signal, str) or len(signal) == 0:
                raise ValueError("rate_rvs keys must be non-empty strings.")
            if not isinstance(rate_rv, RandomVariable):
                raise TypeError(
                    f"rate_rv for signal {signal!r} must be a RandomVariable, "
                    f"got {type(rate_rv).__name__}."
                )

        self.rate_rvs = dict(rate_rvs)
        super().__init__(
            name=name,
            signals=tuple(self.rate_rvs),
            temporal_processes=temporal_processes,
        )

    def _sample_baseline_rates(self) -> Mapping[str, ArrayLike]:
        """Sample one scalar baseline rate from each random variable.

        Returns
        -------
        Mapping[str, ArrayLike]
            Independently sampled scalar rates in signal order.
        """
        return {
            signal: cast(ArrayLike, rate_rv())
            for signal, rate_rv in self.rate_rvs.items()
        }
