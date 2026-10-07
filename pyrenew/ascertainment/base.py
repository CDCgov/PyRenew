# numpydoc ignore=GL08
"""
Base classes for ascertainment models.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping

import jax.nn as jnn
import jax.numpy as jnp
import numpyro
from jax import Array
from jax.scipy.special import logit
from jax.typing import ArrayLike
from numpyro.util import not_jax_tracer

from pyrenew.ascertainment.context import get_ascertainment_value
from pyrenew.latent import TemporalProcess
from pyrenew.metaclass import RandomVariable


class AscertainmentSignal(RandomVariable):
    """
    Accessor for one signal's ascertainment value.

    Users usually do not instantiate this class directly. It is returned by
    ``AscertainmentModel.for_signal(...)`` and passed to an observation process
    as ``ascertainment_rate_rv``. During model execution, the parent
    ``AscertainmentModel`` samples the actual rate once, and this accessor
    retrieves the signal-specific value without creating additional NumPyro
    sample sites.
    """

    def __init__(
        self,
        ascertainment_name: str,
        signal_name: str,
    ) -> None:
        """
        Initialize a signal-specific ascertainment accessor.

        Parameters
        ----------
        ascertainment_name
            Name of the parent ascertainment model.
        signal_name
            Name of the signal to retrieve.
        """
        if not isinstance(ascertainment_name, str) or len(ascertainment_name) == 0:
            raise ValueError(
                "ascertainment_name must be a non-empty string. "
                f"Got {type(ascertainment_name).__name__}: {ascertainment_name!r}"
            )
        if not isinstance(signal_name, str) or len(signal_name) == 0:
            raise ValueError(
                "signal_name must be a non-empty string. "
                f"Got {type(signal_name).__name__}: {signal_name!r}"
            )
        super().__init__(name=f"{ascertainment_name}_{signal_name}")
        self.ascertainment_name = ascertainment_name
        self.signal_name = signal_name

    def sample(self, **kwargs: object) -> ArrayLike:
        """
        Return the sampled ascertainment value for this signal.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments, ignored.

        Returns
        -------
        ArrayLike
            Signal-specific ascertainment value from the active context.
        """
        return get_ascertainment_value(
            ascertainment_name=self.ascertainment_name,
            signal_name=self.signal_name,
        )


class AscertainmentModel(metaclass=ABCMeta):
    """
    An ``AscertainmentModel`` is a component of a PyRenew renewal process
    model. It provides an ascertainment rate for one or more observation signals.

    Each signal has a scalar baseline rate. If a temporal process is specified
    for that signal, the component combines the baseline rate with the temporal
    process to produce a rate trajectory over the model period. Otherwise, the
    baseline rate applies throughout the model period.

    A baseline rate may lie in [0, 1] when no temporal process is configured.
    A signal with a temporal process requires a baseline rate strictly inside
    (0, 1), because the temporal deviation is added on the logit scale.
    At either endpoint, the logit is infinite and finite deviations cannot
    change the rate.

    The model samples one scalar baseline rate per signal and can add an
    optional signal-specific temporal deviation on the logit scale. Subclasses
    implement ``_sample_baseline_rates()`` to define relationships among the
    scalar baselines. The base class owns validation, temporal sampling, and
    the standard deterministic sites.

    Concrete subclasses determine how the baseline rates are specified.
    ``IndependentAscertainment`` specifies each baseline separately,
    ``JointAscertainment`` assigns the baselines a joint distribution, and
    ``RatioLinkedAscertainment`` defines one baseline relative to another.

    Register the component with ``PyrenewBuilder.add_ascertainment(...)``.
    Use ``for_signal(...)`` to obtain the signal-specific accessor passed
    to an observation process as ``ascertainment_rate_rv``. Accessors
    returned by ``for_signal()`` read the final sampled values from
    the active model context and do not sample independently.

    ```python
    ascertainment = JointAscertainment(...)
    builder.add_ascertainment(ascertainment)

    PopulationCounts(
        name="hospital",
        ascertainment_rate_rv=ascertainment.for_signal("hospital"),
        ...
    )
    ```
    """

    def __init__(
        self,
        name: str,
        signals: tuple[str, ...],
        temporal_processes: Mapping[str, TemporalProcess] | None = None,
    ) -> None:
        """
        Initialize an ascertainment model.

        Parameters
        ----------
        name
            A non-empty string identifying the ascertainment model.
        signals
            Unique signal names produced by this model.
        temporal_processes
            Optional temporal processes keyed by signal name. Signals without
            a process retain their scalar baseline rate.

        Raises
        ------
        TypeError
            If ``temporal_processes`` is not a mapping or a configured process
            does not satisfy the ``TemporalProcess`` protocol.
        ValueError
            If a temporal process is configured for an unknown signal.
        """
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError(
                f"name must be a non-empty string. Got {type(name).__name__}: {name!r}"
            )
        if not isinstance(signals, tuple) or len(signals) == 0:
            raise ValueError("signals must be a non-empty tuple of strings.")
        if any(not isinstance(signal, str) or len(signal) == 0 for signal in signals):
            raise ValueError("all signals must be non-empty strings.")
        if len(set(signals)) != len(signals):
            raise ValueError("signals must be unique.")

        self.name = name
        self.signals = signals

        if temporal_processes is None:
            temporal_processes = {}
        if not isinstance(temporal_processes, Mapping):
            raise TypeError(
                "temporal_processes must be a mapping, "
                f"got {type(temporal_processes).__name__}."
            )

        expected_signals = set(signals)
        unknown_signals = tuple(
            signal for signal in temporal_processes if signal not in expected_signals
        )
        if unknown_signals:
            raise ValueError(
                f"temporal_processes contains unknown signals {unknown_signals} for "
                f"ascertainment model {name!r}. Available signals: {signals}."
            )

        ordered_processes: dict[str, TemporalProcess] = {}
        for signal in signals:
            if signal not in temporal_processes:
                continue
            process = temporal_processes[signal]
            if not isinstance(process, TemporalProcess):
                raise TypeError(
                    f"temporal process for signal {signal!r} must satisfy the "
                    f"TemporalProcess protocol, got {type(process).__name__}."
                )
            ordered_processes[signal] = process
        self.temporal_processes = ordered_processes

    def for_signal(self, signal_name: str) -> AscertainmentSignal:
        """
        Return an observation-process accessor for one signal.

        Parameters
        ----------
        signal_name
            Name of the signal produced by this ascertainment model. This name
            should match the signal name used when the ascertainment model was
            constructed. It does not have to match the observation process name,
            but using the same name usually makes model specifications easier
            to read.

        Returns
        -------
        AscertainmentSignal
            RandomVariable-compatible accessor for the signal's sampled
            ascertainment rate.

        Raises
        ------
        ValueError
            If ``signal_name`` is not produced by this model.
        """
        if signal_name not in self.signals:
            raise ValueError(
                f"Unknown signal {signal_name!r} for ascertainment model "
                f"{self.name!r}. Available signals: {self.signals}."
            )
        return AscertainmentSignal(
            ascertainment_name=self.name,
            signal_name=signal_name,
        )

    def requires_calendar_anchor(self) -> bool:
        """Return whether any temporal process requires a calendar anchor.

        Returns
        -------
        bool
            ``True`` when the model needs a calendar anchor; otherwise
            ``False``.
        """
        return any(
            getattr(process, "requires_calendar_anchor", False)
            for process in self.temporal_processes.values()
        )

    def _validate_baseline_rates(
        self,
        baseline_rates: Mapping[str, ArrayLike],
    ) -> dict[str, Array]:
        """Validate and convert scalar baseline ascertainment rates.

        Parameters
        ----------
        baseline_rates
            Sampled scalar baseline rates keyed by signal name.

        Returns
        -------
        dict[str, Array]
            Validated scalar arrays in signal order.

        Raises
        ------
        TypeError
            If ``baseline_rates`` is not a mapping.
        ValueError
            If signal names do not match, a baseline is not scalar, or a
            concrete baseline is non-finite or outside its allowed interval.
        """
        if not isinstance(baseline_rates, Mapping):
            raise TypeError(
                f"Ascertainment model {self.name!r} must return baseline rates "
                f"as a mapping, got {type(baseline_rates).__name__}."
            )

        expected_signals = set(self.signals)
        actual_signals = set(baseline_rates)
        if actual_signals != expected_signals:
            missing = tuple(
                signal for signal in self.signals if signal not in baseline_rates
            )
            extra = tuple(
                signal for signal in baseline_rates if signal not in expected_signals
            )
            raise ValueError(
                f"Ascertainment model {self.name!r} must return baseline rates for "
                f"exactly signals {self.signals}. Missing: {missing}. Extra: {extra}."
            )

        validated_rates: dict[str, Array] = {}
        for signal in self.signals:
            baseline_rate = jnp.asarray(baseline_rates[signal])
            if baseline_rate.shape != ():
                raise ValueError(
                    f"Ascertainment model {self.name!r}, signal {signal!r}, "
                    f"returned baseline shape {baseline_rate.shape}; required shape "
                    "is ()."
                )

            if signal in self.temporal_processes:
                invalid_rate = (
                    ~jnp.isfinite(baseline_rate)
                    | (baseline_rate <= 0)
                    | (baseline_rate >= 1)
                )
                allowed_interval = "strictly inside (0, 1)"
            else:
                invalid_rate = (
                    ~jnp.isfinite(baseline_rate)
                    | (baseline_rate < 0)
                    | (baseline_rate > 1)
                )
                allowed_interval = "inside [0, 1]"

            if not_jax_tracer(invalid_rate) and bool(invalid_rate):
                raise ValueError(
                    f"Ascertainment model {self.name!r} requires the baseline for "
                    f"signal {signal!r} to be finite and {allowed_interval}; got "
                    f"{baseline_rate}."
                )
            validated_rates[signal] = baseline_rate

        return validated_rates

    def sample(
        self,
        n_timepoints: int,
        first_day_dow: int | None = None,
    ) -> Mapping[str, ArrayLike]:
        """Sample baseline rates and optional temporal rate trajectories.

        Parameters
        ----------
        n_timepoints
            Positive number of timepoints on the shared model axis.
        first_day_dow
            Day of week for the first model-axis timepoint. Calendar-aligned
            temporal processes use this value.

        Returns
        -------
        Mapping[str, ArrayLike]
            Final scalar rates or full-axis trajectories in signal order.

        Raises
        ------
        TypeError
            If ``n_timepoints`` is not a built-in integer.
        ValueError
            If ``n_timepoints`` is less than one, baseline rates are invalid,
            or a temporal process returns an unexpected shape.
        """
        if type(n_timepoints) is not int:
            raise TypeError(
                "n_timepoints must be a positive integer, "
                f"got {type(n_timepoints).__name__}."
            )
        if n_timepoints < 1:
            raise ValueError(
                f"n_timepoints must be a positive integer, got {n_timepoints}."
            )

        baseline_rates = self._validate_baseline_rates(self._sample_baseline_rates())

        result: dict[str, ArrayLike] = {}
        for signal in self.signals:
            baseline_rate = baseline_rates[signal]
            numpyro.deterministic(
                f"{self.name}_baseline_{signal}",
                baseline_rate,
            )

            process = self.temporal_processes.get(signal)
            if process is None:
                rate = baseline_rate
            else:
                deviation = jnp.asarray(
                    process.sample(
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
                        f"Ascertainment model {self.name!r}, signal {signal!r}, "
                        f"received temporal-process shape {deviation.shape}; "
                        f"required shape is {required_shape}."
                    )
                deviation = jnp.squeeze(deviation, axis=-1)
                rate = jnn.sigmoid(logit(baseline_rate) + deviation)

            numpyro.deterministic(f"{self.name}_{signal}", rate)
            result[signal] = rate

        return result

    def validate_sampled_values(
        self,
        values: Mapping[str, ArrayLike],
        n_timepoints: int,
    ) -> None:
        """Validate sampled signal names and output shapes.

        Parameters
        ----------
        values
            Sampled ascertainment values keyed by signal name.
        n_timepoints
            Length of the shared model time axis.

        Raises
        ------
        TypeError
            If ``values`` is not a mapping.
        ValueError
            If signal names do not match this model or a value is neither a
            scalar nor a full-axis trajectory.
        """
        if not isinstance(values, Mapping):
            raise TypeError(
                f"Ascertainment model {self.name!r} must return a mapping, "
                f"got {type(values).__name__}."
            )

        actual_signals = set(values)
        expected_signals = set(self.signals)
        if actual_signals != expected_signals:
            missing = tuple(signal for signal in self.signals if signal not in values)
            extra = tuple(signal for signal in values if signal not in expected_signals)
            raise ValueError(
                f"Ascertainment model {self.name!r} must return exactly signals "
                f"{self.signals}. Missing: {missing}. Extra: {extra}."
            )

        allowed_shapes = ((), (n_timepoints,))
        for signal in self.signals:
            actual_shape = jnp.asarray(values[signal]).shape
            if actual_shape not in allowed_shapes:
                raise ValueError(
                    f"Ascertainment model {self.name!r}, signal {signal!r}, "
                    f"returned shape {actual_shape}; allowed shapes are () and "
                    f"({n_timepoints},)."
                )

    @abstractmethod
    def _sample_baseline_rates(self) -> Mapping[str, ArrayLike]:
        """Sample one scalar baseline ascertainment rate per signal.

        Returns
        -------
        Mapping[str, ArrayLike]
            Scalar baseline rates keyed by the model's signal names.

        Notes
        -----
        Custom subclasses should implement only baseline sampling here. The
        base ``sample()`` method validates baselines, applies any configured
        temporal processes, and records the standard deterministic sites.
        """
        pass  # pragma: no cover
