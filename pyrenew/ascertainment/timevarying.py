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

    For each signal, a temporal process is sampled as a logit-scale deviation
    from an explicit natural-scale baseline ascertainment rate:

    ```text
    eta_j(t) = logit(baseline_rate_j) + z_j(t)
    ascertainment_rate_j(t) = sigmoid(eta_j(t))
    ```

    ``baseline_rate_j`` is the baseline ascertainment probability for signal
    ``j`` and ``z_j(t)`` is the signal's temporal deviation. The returned values
    are rate trajectories on the model time axis. When used with
    ``PopulationCounts``, these rates multiply latent incidence before delay
    convolution, so they are interpreted as incident-time ascertainment
    probabilities.
    """

    def __init__(
        self,
        name: str,
        processes: Mapping[str, TemporalProcess],
        baseline_rates: Mapping[str, ArrayLike],
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
        baseline_rates
            Mapping from signal name to scalar natural-scale baseline
            ascertainment rate. Must contain exactly one entry for each signal
            in ``processes``. Values must be probabilities in ``(0, 1)``.
        """
        if not isinstance(processes, Mapping) or len(processes) == 0:
            raise ValueError("processes must be a non-empty mapping.")
        if not isinstance(baseline_rates, Mapping):
            raise TypeError("baseline_rates must be a mapping.")
        signals = tuple(processes.keys())
        super().__init__(name=name, signals=signals)

        self.processes = dict(processes)
        self.baseline_rates = dict(baseline_rates)
        self.baseline_logits = {}
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """
        Validate process and baseline-rate parameters.
        """
        for signal, process in self.processes.items():
            if not isinstance(process, TemporalProcess):
                raise TypeError(
                    f"process for signal {signal!r} must satisfy "
                    "the TemporalProcess protocol."
                )

        baseline_signals = set(self.baseline_rates)
        process_signals = set(self.signals)
        unknown_baselines = baseline_signals - process_signals
        missing_baselines = process_signals - baseline_signals
        if unknown_baselines or missing_baselines:
            raise ValueError(
                "baseline_rates must contain exactly the same signal names as "
                "processes. "
                f"Missing: {tuple(sorted(missing_baselines))}. "
                f"Unknown: {tuple(sorted(unknown_baselines))}."
            )

        for signal, baseline_rate in self.baseline_rates.items():
            baseline_rate_array = jnp.asarray(baseline_rate)
            if baseline_rate_array.shape != ():
                raise ValueError(
                    f"baseline rate for signal {signal!r} must be scalar, "
                    f"got shape {baseline_rate_array.shape}."
                )
            if baseline_rate_array <= 0 or baseline_rate_array >= 1:
                raise ValueError(
                    f"baseline rate for signal {signal!r} must be in (0, 1), "
                    f"got {baseline_rate_array}."
                )
            self.baseline_rates[signal] = baseline_rate_array
            self.baseline_logits[signal] = jnp.log(baseline_rate_array) - jnp.log1p(
                -baseline_rate_array
            )

    def _baseline_logit(self, signal: str) -> ArrayLike:
        """
        Return the baseline logit ascertainment level for a signal.

        Returns
        -------
        ArrayLike
            Baseline logit ascertainment level for ``signal``.
        """
        return self.baseline_logits[signal]

    def calendar_anchor_requirements(self) -> tuple[str, ...]:
        """
        Return signals whose temporal process requires a calendar anchor.

        Returns
        -------
        tuple of str
            Signal names backed by calendar-aligned temporal processes.
        """
        return tuple(
            signal
            for signal, process in self.processes.items()
            if getattr(process, "requires_calendar_anchor", False)
        )

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
            eta = eta_deviation + self._baseline_logit(signal)
            rate = jnn.sigmoid(eta)
            numpyro.deterministic(f"{self.name}_{signal}", rate)
            result[signal] = rate

        return result
