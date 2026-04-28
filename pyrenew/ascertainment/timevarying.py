# numpydoc ignore=GL08
"""
Time-varying ascertainment models.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.nn as jnn
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike

from pyrenew.ascertainment.base import AscertainmentModel
from pyrenew.latent import TemporalProcess


class TimeVaryingAscertainment(AscertainmentModel):
    """
    Time-varying ascertainment rates for one or more count signals.

    This model represents probabilities that latent incidence is observed in
    one or more data streams, with those probabilities allowed to change over
    the model time axis. For example, the probability that an infection is
    observed as an emergency department visit may vary if care-seeking,
    testing, or reporting practices change over time.

    For each signal, a temporal process is sampled on the logit scale and
    transformed to a probability:

    ```text
    eta_j(t) = loc_j + z_j(t)
    ascertainment_rate_j(t) = sigmoid(eta_j(t))
    ```

    ``loc_j`` is the baseline logit ascertainment level for signal ``j`` and
    ``z_j(t)`` is the signal's temporal deviation. The returned values are rate
    trajectories on the model time axis. When used with ``PopulationCounts``,
    these rates multiply latent incidence before delay convolution, so they
    are interpreted as incident-time ascertainment probabilities.
    """

    def __init__(
        self,
        name: str,
        processes: Mapping[str, TemporalProcess],
        locs: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        """
        Initialize a time-varying ascertainment model.

        Parameters
        ----------
        name
            Name of the ascertainment model.
        processes
            Mapping from signal name to temporal process. Each process is
            sampled on the logit scale and must satisfy the ``TemporalProcess``
            protocol. Wrappers such as ``WeeklyTemporalProcess`` can be used
            when the ascertainment rate should be estimated at a coarser
            cadence and broadcast to the model time axis.
        locs
            Optional mapping from signal name to scalar logit-scale location.
            Signals not present in ``locs`` use ``0.0``, corresponding to a
            baseline rate of 0.5 before temporal deviations.
        """
        if not isinstance(processes, Mapping) or len(processes) == 0:
            raise ValueError("processes must be a non-empty mapping.")
        signals = tuple(processes.keys())
        super().__init__(name=name, signals=signals)

        self.processes = dict(processes)
        self.locs = {} if locs is None else dict(locs)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """
        Validate process and location parameters.
        """
        for signal, process in self.processes.items():
            if not isinstance(process, TemporalProcess):
                raise TypeError(
                    f"process for signal {signal!r} must satisfy "
                    "the TemporalProcess protocol."
                )

        unknown_locs = set(self.locs) - set(self.signals)
        if unknown_locs:
            raise ValueError(
                f"locs contains unknown signal names: {tuple(sorted(unknown_locs))}."
            )

        for signal, loc in self.locs.items():
            loc_array = jnp.asarray(loc)
            if loc_array.shape != ():
                raise ValueError(
                    f"loc for signal {signal!r} must be scalar, "
                    f"got shape {loc_array.shape}."
                )
            self.locs[signal] = loc_array

    def _loc(self, signal: str) -> ArrayLike:
        """
        Return the baseline logit ascertainment level for a signal.

        Returns
        -------
        ArrayLike
            Baseline logit ascertainment level for ``signal``.
        """
        return self.locs.get(signal, jnp.asarray(0.0))

    def sample(
        self,
        n_timepoints: int,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> Mapping[str, ArrayLike]:
        """
        Sample signal-specific ascertainment-rate trajectories.

        Parameters
        ----------
        n_timepoints
            Number of time points on the model axis.
        first_day_dow
            Day-of-week index for element 0 of the model axis. Calendar-aware
            temporal processes, such as ``WeeklyTemporalProcess``, require this
            value to align their coarse time steps to calendar weeks.
        **kwargs
            Additional model-context arguments, ignored.

        Returns
        -------
        Mapping[str, ArrayLike]
            Mapping from signal name to sampled ascertainment-rate trajectory.
            Each trajectory has shape ``(n_timepoints,)``.
        """
        result = {}
        for signal, process in self.processes.items():
            eta_deviation = process.sample(
                n_timepoints=n_timepoints,
                initial_value=0.0,
                n_processes=1,
                name_prefix=f"{self.name}_{signal}",
                first_day_dow=first_day_dow,
            )
            eta_deviation = jnp.squeeze(eta_deviation, axis=-1)
            eta = eta_deviation + self._loc(signal)
            rate = jnn.sigmoid(eta)
            numpyro.deterministic(f"{self.name}_{signal}", rate)
            result[signal] = rate

        return result
