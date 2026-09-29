# numpydoc ignore=GL08
"""
Time-varying ascertainment models.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.nn as jnn
import jax.numpy as jnp
import numpyro
from jax.scipy.special import logit
from jax.typing import ArrayLike
from numpyro.util import not_jax_tracer

from pyrenew.ascertainment.base import AscertainmentModel
from pyrenew.latent import TemporalProcess


class TimeVaryingAscertainment(AscertainmentModel):
    """
    Add independent temporal deviations to related ascertainment baselines.

    The wrapped baseline model samples one scalar rate per signal. Each scalar
    is converted to the logit scale, combined with a signal-specific temporal
    deviation, and returned as a probability trajectory on the shared model
    time axis. Baseline relationships therefore describe the reference rates,
    not a pointwise constraint on the resulting trajectories.

    Register only this outer model with ``PyrenewBuilder``. Use accessors from
    this model, such as ``ascertainment.for_signal("hospital")``, in count
    observation processes.
    """

    def __init__(
        self,
        name: str,
        baseline_model: AscertainmentModel,
        processes: Mapping[str, TemporalProcess],
    ) -> None:
        """
        Initialize time-varying ascertainment.

        Parameters
        ----------
        name
            Name of the outer ascertainment model. It must differ from the
            wrapped baseline model's name.
        baseline_model
            Ascertainment model that samples one scalar reference rate for
            each signal.
        processes
            One temporal process per baseline signal. Each process generates
            an independent logit-scale deviation trajectory.

        Raises
        ------
        TypeError
            If the baseline or temporal processes have invalid types.
        ValueError
            If names conflict or process signal names do not match the
            baseline signals.
        """
        if not isinstance(baseline_model, AscertainmentModel):
            raise TypeError(
                "baseline_model must be an AscertainmentModel, "
                f"got {type(baseline_model).__name__}."
            )
        if not isinstance(processes, Mapping):
            raise TypeError(
                f"processes must be a mapping, got {type(processes).__name__}."
            )
        if name == baseline_model.name:
            raise ValueError(
                "Time-varying ascertainment and its baseline model must have "
                f"different names; both were {name!r}."
            )

        super().__init__(name=name, signals=baseline_model.signals)
        expected_signals = set(self.signals)
        process_signals = set(processes)
        if process_signals != expected_signals:
            missing = tuple(
                signal for signal in self.signals if signal not in processes
            )
            extra = tuple(
                signal for signal in processes if signal not in expected_signals
            )
            raise ValueError(
                "processes must contain exactly the baseline model signals "
                f"{self.signals}. Missing: {missing}. Extra: {extra}."
            )

        ordered_processes: dict[str, TemporalProcess] = {}
        for signal in self.signals:
            process = processes[signal]
            if not isinstance(process, TemporalProcess):
                raise TypeError(
                    f"process for signal {signal!r} must satisfy the "
                    f"TemporalProcess protocol, got {type(process).__name__}."
                )
            ordered_processes[signal] = process

        self.baseline_model = baseline_model
        self.processes = ordered_processes

    def requires_calendar_anchor(self) -> bool:
        """Return whether the baseline or any temporal process needs a date.

        Returns
        -------
        bool
            ``True`` when sampling needs the model-axis day-of-week; otherwise
            ``False``.
        """
        return self.baseline_model.requires_calendar_anchor() or any(
            getattr(process, "requires_calendar_anchor", False)
            for process in self.processes.values()
        )

    def sample(
        self,
        n_timepoints: int,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> Mapping[str, ArrayLike]:
        """
        Sample full-axis ascertainment-rate trajectories.

        Parameters
        ----------
        n_timepoints
            Number of timepoints on the shared model axis.
        first_day_dow
            Day-of-week for the first model-axis timepoint. Required by
            calendar-aligned temporal processes.
        **kwargs
            Additional model context, ignored.

        Returns
        -------
        Mapping[str, ArrayLike]
            One probability trajectory of shape ``(n_timepoints,)`` per
            signal.

        Raises
        ------
        ValueError
            If the baseline does not return scalar probabilities in ``(0, 1)``
            or a temporal process returns an unexpected shape.
        """
        baseline_values = self.baseline_model.sample(
            n_timepoints=n_timepoints,
            first_day_dow=first_day_dow,
        )
        self.baseline_model.validate_sampled_values(
            baseline_values,
            n_timepoints=n_timepoints,
        )

        scalar_baselines: dict[str, ArrayLike] = {}
        for signal in self.signals:
            baseline_value = jnp.asarray(baseline_values[signal])
            if baseline_value.shape != ():
                raise ValueError(
                    f"Time-varying ascertainment model {self.name!r} requires "
                    f"a scalar baseline for signal {signal!r}; got shape "
                    f"{baseline_value.shape}, required shape ()."
                )
            invalid_baseline = (
                ~jnp.isfinite(baseline_value)
                | (baseline_value <= 0)
                | (baseline_value >= 1)
            )
            if not_jax_tracer(invalid_baseline) and bool(invalid_baseline):
                raise ValueError(
                    f"Time-varying ascertainment model {self.name!r} requires "
                    f"the baseline for signal {signal!r} to be finite and "
                    f"strictly inside (0, 1); got {baseline_value}."
                )
            scalar_baselines[signal] = baseline_value

        result: dict[str, ArrayLike] = {}
        for signal in self.signals:
            deviation = jnp.asarray(
                self.processes[signal].sample(
                    n_timepoints=n_timepoints,
                    initial_value=0.0,
                    n_processes=1,
                    name_prefix=f"{self.name}_{signal}",
                    first_day_dow=first_day_dow,
                )
            )
            required_shape = (n_timepoints, 1)
            if deviation.shape != required_shape:
                raise ValueError(
                    f"Time-varying ascertainment model {self.name!r}, signal "
                    f"{signal!r}, received temporal-process shape "
                    f"{deviation.shape}; required shape is {required_shape}."
                )
            deviation = jnp.squeeze(deviation, axis=-1)
            rate = jnn.sigmoid(logit(scalar_baselines[signal]) + deviation)
            numpyro.deterministic(f"{self.name}_{signal}", rate)
            result[signal] = rate

        return result
