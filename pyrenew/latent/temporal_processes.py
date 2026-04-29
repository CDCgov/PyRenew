# numpydoc ignore=GL08
"""
Temporal processes for latent infection models.

Provides time-series processes for modeling Rt dynamics and subpopulation
deviations in hierarchical infection models. All processes return 2D arrays
of shape (n_timepoints, n_processes) through a unified ``TemporalProcess``
protocol.

**Relationship to pyrenew.process:**

This module provides **high-level, domain-specific wrappers** around the
low-level building blocks in [pyrenew.process][]. The key differences:

| Aspect | ``pyrenew.process`` | ``pyrenew.latent.temporal_processes`` |
| --- | --- | --- |
| Abstraction level | Low-level composable primitives | High-level domain-specific API |
| Interface | Varied signatures per class | Unified ``TemporalProcess`` protocol |
| Target use | General time-series modeling | Rt dynamics, hierarchical infections |
| Vectorization | Caller manages array shapes | Automatic via ``n_processes`` parameter |
| Validation | Minimal constraints | Validates positive innovation_sd |

**When to use which:**

- Use ``pyrenew.process`` classes (``ARProcess``, ``DifferencedProcess``,
  ``RandomWalk``) when building novel statistical models or when you need
  fine-grained control over array shapes and numpyro sampling semantics.

- Use this module's classes (``AR1``, ``DifferencedAR1``, ``RandomWalk``)
  when modeling Rt trajectories in hierarchical infection models. These
  provide a consistent interface, automatic vectorization, and enforce
  epidemiologically-sensible constraints.

**Temporal processes provided:**

- ``AR1``: Autoregressive process with mean reversion. Keeps Rt bounded
  near a baseline. Wraps [pyrenew.process.ARProcess][].
- ``DifferencedAR1``: AR(1) on first differences. Allows persistent trends
  while stabilizing the growth rate. Wraps [pyrenew.process.DifferencedProcess][].
- ``RandomWalk``: No mean reversion. Rt can drift without bound.
  Wraps [pyrenew.process.RandomWalk][].
- ``StepwiseTemporalProcess``: Wrapper that parameterizes any inner
  ``TemporalProcess`` at a coarser model-index cadence and broadcasts to the
  per-timepoint scale by repetition.
- ``WeeklyTemporalProcess``: Wrapper that parameterizes any inner
  ``TemporalProcess`` at a calendar-week cadence and broadcasts to the
  per-timepoint scale by repetition.

All implementations satisfy the ``TemporalProcess`` protocol and can be
used interchangeably in hierarchical infection models.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from pyrenew.process import ARProcess, DifferencedProcess
from pyrenew.process.randomwalk import RandomWalk as ProcessRandomWalk
from pyrenew.randomvariable import DistributionalVariable
from pyrenew.time import validate_dow, weekly_to_daily


@runtime_checkable
class TemporalProcess(Protocol):
    """
    Protocol for temporal processes generating time-varying parameters.

    Used for jurisdiction-level Rt dynamics, subpopulation deviations, or
    allocation trajectories. All processes return 2D arrays of shape
    (n_timepoints, n_processes) for consistent handling.

    Attributes
    ----------
    step_size : int
        Number of consecutive timepoints that share the same sampled value.
        Defaults to ``1`` for the standard processes (one independent sample
        per timepoint). Wrapper processes like ``StepwiseTemporalProcess`` and
        ``WeeklyTemporalProcess`` expose a larger value so that model builders
        can inspect R(t) parametrization cadence.
    """

    step_size: int

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int = 1,
        name_prefix: str = "temporal",
        *,
        first_day_dow: int | None = None,
    ) -> ArrayLike:
        """
        Sample temporal trajectory or trajectories.

        Parameters
        ----------
        n_timepoints
            Number of time points to generate
        initial_value
            Initial value(s) for the process(es).
            Scalar (broadcast to all processes) or array of shape (n_processes,).
            Defaults to 0.0.
        n_processes
            Number of parallel processes.
        name_prefix
            Prefix for numpyro sample site names to avoid collisions
        first_day_dow
            Day of week for element 0 of the shared model time axis
            (0=Monday, ..., 6=Sunday). Standard temporal processes ignore
            this value; calendar-aligned wrappers may use it.

        Returns
        -------
        ArrayLike
            Trajectories of shape (n_timepoints, n_processes)
        """
        ...


class AR1(TemporalProcess):
    """
    AR(1) process.

    Each value depends on the previous value plus noise, with reversion
    toward a mean level. Keeps Rt bounded near a baseline — values that
    drift away are "pulled back" over time.

    This class wraps [pyrenew.process.ARProcess][] with a simplified,
    protocol-compliant interface that handles vectorization automatically.

    Parameters
    ----------
    autoreg
        Autoregressive coefficient. For stationarity, |autoreg| < 1, but
        this is not enforced (use priors to constrain if needed).
    innovation_sd
        Standard deviation of noise at each time step. Larger values produce
        more volatile trajectories; smaller values produce smoother ones.
    """

    step_size: int = 1

    def __init__(self, autoreg: float, innovation_sd: float = 1.0) -> None:
        """
        Initialize AR(1) process.

        Parameters
        ----------
        autoreg
            Autoregressive coefficient. For stationarity, |autoreg| < 1,
            but this is not enforced (use priors to constrain if needed).
        innovation_sd
            Standard deviation of innovations

        Raises
        ------
        ValueError
            If innovation_sd <= 0
        """
        if innovation_sd <= 0:
            raise ValueError(f"innovation_sd must be positive, got {innovation_sd}")
        self.autoreg = autoreg
        self.innovation_sd = innovation_sd
        self.ar_process = ARProcess(name="ar1")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AR1(autoreg={self.autoreg}, innovation_sd={self.innovation_sd})"

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int = 1,
        name_prefix: str = "ar1",
        *,
        first_day_dow: int | None = None,
    ) -> ArrayLike:
        """
        Sample AR(1) trajectory or trajectories.

        Parameters
        ----------
        n_timepoints
            Number of time points to generate
        initial_value
            Initial value(s). Defaults to 0.0.
        n_processes
            Number of parallel processes.
        name_prefix
            Prefix for numpyro sample sites
        first_day_dow
            Unused. See [pyrenew.latent.TemporalProcess][].

        Returns
        -------
        ArrayLike
            Trajectories of shape (n_timepoints, n_processes)
        """
        if initial_value is None:
            initial_value = jnp.zeros(n_processes)
        elif jnp.isscalar(initial_value):
            initial_value = jnp.full(n_processes, initial_value)

        stationary_sd = self.innovation_sd / jnp.sqrt(1 - self.autoreg**2)

        with numpyro.plate(f"{name_prefix}_init_plate", n_processes):
            init_states = numpyro.sample(
                f"{name_prefix}_init",
                dist.Normal(initial_value, stationary_sd),
            )

        trajectories = self.ar_process(
            n=n_timepoints,
            init_vals=init_states[jnp.newaxis, :],
            autoreg=jnp.full((1, n_processes), self.autoreg),
            noise_sd=self.innovation_sd,
            noise_name=f"{name_prefix}_noise",
        )

        return trajectories


class DifferencedAR1(TemporalProcess):
    """
    AR(1) process on first differences.

    Each *change* in value depends on the previous change plus noise, with
    the rate of change reverting toward a mean. Unlike AR(1), this allows
    Rt to trend persistently upward or downward while the growth rate stabilizes.

    This class wraps [pyrenew.process.DifferencedProcess][] with
    [pyrenew.process.ARProcess][] as the fundamental process, providing
    a simplified, protocol-compliant interface.

    Parameters
    ----------
    autoreg
        Autoregressive coefficient for differences. For stationarity,
        |autoreg| < 1, but this is not enforced (use priors to constrain
        if needed).
    innovation_sd
        Standard deviation of noise added to changes. Larger values produce
        more erratic growth rates; smaller values produce smoother trends.
    """

    step_size: int = 1

    def __init__(self, autoreg: float, innovation_sd: float = 1.0) -> None:
        """
        Initialize differenced AR(1) process.

        Parameters
        ----------
        autoreg
            Autoregressive coefficient for differences. For stationarity,
            |autoreg| < 1, but this is not enforced (use priors to constrain
            if needed).
        innovation_sd
            Standard deviation of innovations

        Raises
        ------
        ValueError
            If innovation_sd <= 0
        """
        if innovation_sd <= 0:
            raise ValueError(f"innovation_sd must be positive, got {innovation_sd}")
        self.autoreg = autoreg
        self.innovation_sd = innovation_sd
        self.process = DifferencedProcess(
            name="diff_ar1",
            fundamental_process=ARProcess(name="diff_ar1_fundamental"),
            differencing_order=1,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DifferencedAR1(autoreg={self.autoreg}, innovation_sd={self.innovation_sd})"

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int = 1,
        name_prefix: str = "diff_ar1",
        *,
        first_day_dow: int | None = None,
    ) -> ArrayLike:
        """
        Sample differenced AR(1) trajectory or trajectories.

        Parameters
        ----------
        n_timepoints
            Number of time points to generate
        initial_value
            Initial value(s). Defaults to 0.0.
        n_processes
            Number of parallel processes.
        name_prefix
            Prefix for numpyro sample sites
        first_day_dow
            Unused. See [pyrenew.latent.TemporalProcess][].

        Returns
        -------
        ArrayLike
            Trajectories of shape (n_timepoints, n_processes)
        """
        if initial_value is None:
            initial_value = jnp.zeros(n_processes)
        elif jnp.isscalar(initial_value):
            initial_value = jnp.full(n_processes, initial_value)

        stationary_sd = self.innovation_sd / jnp.sqrt(1 - self.autoreg**2)

        with numpyro.plate(f"{name_prefix}_init_rate_plate", n_processes):
            init_rates = numpyro.sample(
                f"{name_prefix}_init_rate",
                dist.Normal(0, stationary_sd),
            )

        trajectories = self.process(
            n=n_timepoints,
            init_vals=initial_value[jnp.newaxis, :],
            autoreg=jnp.full((1, n_processes), self.autoreg),
            noise_sd=self.innovation_sd,
            fundamental_process_init_vals=init_rates[jnp.newaxis, :],
            noise_name=f"{name_prefix}_noise",
        )

        return trajectories


class RandomWalk(TemporalProcess):
    """
    Random walk process for log(Rt).

    Each value equals the previous value plus noise, with no reversion
    toward a mean. Allows Rt to drift without bound — suitable when you
    have no prior expectation that Rt will return to a baseline.

    This class wraps [pyrenew.process.RandomWalk][] with a simplified,
    protocol-compliant interface that handles vectorization automatically.

    Parameters
    ----------
    innovation_sd
        Standard deviation of noise at each time step. Larger values produce
        faster drift; smaller values produce more gradual changes.

    Notes
    -----
    Unlike AR(1), variance grows over time — the process can wander arbitrarily
    far from its starting point. For long time horizons, consider AR(1) if you
    want Rt to stay bounded near a baseline.

    For non-centered parameterization (to avoid funnel problems in inference),
    apply ``LocScaleReparam(centered=0)`` to the step sample site
    (``{name_prefix}_step``) via ``numpyro.handlers.reparam``.
    """

    step_size: int = 1

    def __init__(self, innovation_sd: float = 1.0) -> None:
        """
        Initialize random walk process.

        Parameters
        ----------
        innovation_sd
            Standard deviation of innovations

        Raises
        ------
        ValueError
            If innovation_sd <= 0
        """
        if innovation_sd <= 0:
            raise ValueError(f"innovation_sd must be positive, got {innovation_sd}")
        self.innovation_sd = innovation_sd

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RandomWalk(innovation_sd={self.innovation_sd})"

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int = 1,
        name_prefix: str = "rw",
        *,
        first_day_dow: int | None = None,
    ) -> ArrayLike:
        """
        Sample random walk trajectory or trajectories.

        Parameters
        ----------
        n_timepoints
            Number of time points to generate
        initial_value
            Initial value(s). Defaults to 0.0.
        n_processes
            Number of parallel processes.
        name_prefix
            Prefix for numpyro sample sites
        first_day_dow
            Unused. See [pyrenew.latent.TemporalProcess][].

        Returns
        -------
        ArrayLike
            Trajectories of shape (n_timepoints, n_processes)
        """
        if initial_value is None:
            initial_value = jnp.zeros(n_processes)
        elif jnp.isscalar(initial_value):
            initial_value = jnp.full(n_processes, initial_value)

        rw = ProcessRandomWalk(
            name=f"{name_prefix}_random_walk",
            step_rv=DistributionalVariable(
                name=f"{name_prefix}_step",
                distribution=dist.Normal(
                    jnp.zeros(n_processes),
                    self.innovation_sd,
                ),
            ),
        )

        return rw.sample(
            init_vals=initial_value[jnp.newaxis, :],
            n=n_timepoints,
        )


class StepwiseTemporalProcess(TemporalProcess):
    """
    Parameterize an inner temporal process at a coarser cadence and
    broadcast to the per-timepoint scale by model-index repetition.

    Each ``step_size`` consecutive output timepoints share a single sampled
    value from the inner process. Use when a parameter should be estimated at
    a coarser cadence while downstream model components still need one value
    per evaluation timepoint. Blocks always start at output index 0.

    Parameters
    ----------
    inner
        Inner ``TemporalProcess`` that generates the coarse-scale
        trajectory. Must satisfy the ``TemporalProcess`` protocol.
    step_size
        Number of per-timepoint units that share each inner sample.
        Must be a positive integer.

    Raises
    ------
    ValueError
        If ``step_size`` is not a positive integer.
    """

    def __init__(
        self,
        inner: TemporalProcess,
        step_size: int,
    ) -> None:
        """
        Initialize stepwise temporal process.

        Parameters
        ----------
        inner
            Inner ``TemporalProcess`` that generates the coarse trajectory.
        step_size
            Number of per-timepoint units that share each inner sample.

        Raises
        ------
        ValueError
            If ``step_size`` is not a positive integer.
        """
        if not isinstance(step_size, int) or step_size < 1:
            raise ValueError(f"step_size must be a positive integer, got {step_size!r}")
        self.inner = inner
        self.step_size = step_size

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"StepwiseTemporalProcess(inner={self.inner!r}, step_size={self.step_size})"
        )

    def _resolve_n_coarse(self, n_timepoints: int) -> int:
        """
        Return the number of inner-process samples needed.

        Returns
        -------
        int
            Number of coarse samples required to cover ``n_timepoints`` under
            model-index-aligned repetition.
        """
        return (n_timepoints + self.step_size - 1) // self.step_size

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int = 1,
        name_prefix: str = "stepwise",
        *,
        first_day_dow: int | None = None,
    ) -> ArrayLike:
        """
        Sample coarse trajectory from inner process and broadcast.

        Computes the number of coarse time steps needed for the requested
        length, samples the inner process at that cadence, then broadcasts each
        coarse value to the per-timepoint axis and trims to ``n_timepoints``.
        The returned value always has one row per evaluation timepoint,
        regardless of the inner parameter cadence. The coarse
        trajectory is recorded as a NumPyro deterministic site named
        ``"{name_prefix}_coarse"``.

        Parameters
        ----------
        n_timepoints
            Number of per-timepoint outputs to produce.
        initial_value
            Initial value(s) for the inner process. Defaults to 0.0.
        n_processes
            Number of parallel processes.
        name_prefix
            Prefix for numpyro sample sites; forwarded to the inner process.
        first_day_dow
            Ignored. Accepted for compatibility with the
            ``TemporalProcess`` protocol.

        Returns
        -------
        ArrayLike
            Trajectories of shape ``(n_timepoints, n_processes)``, constant
            within each block of ``step_size`` consecutive rows.
        """
        n_steps = self._resolve_n_coarse(n_timepoints)
        coarse = self.inner.sample(
            n_timepoints=n_steps,
            initial_value=initial_value,
            n_processes=n_processes,
            name_prefix=name_prefix,
        )
        numpyro.deterministic(f"{name_prefix}_coarse", coarse)
        return jnp.repeat(coarse, repeats=self.step_size, axis=0)[:n_timepoints]


class WeeklyTemporalProcess(TemporalProcess):
    """
    Parameterize an inner temporal process at a calendar-week cadence.

    Each output timepoint in the same calendar week shares a single sampled
    value from the inner process. Use when a parameter should be estimated once
    per week while downstream model components still need one value per
    evaluation timepoint. For example, a weekly-parameterized R(t) process can
    return daily R(t) values for a daily deterministic renewal equation.

    Parameters
    ----------
    inner
        Inner ``TemporalProcess`` that generates the weekly trajectory.
        Must satisfy the ``TemporalProcess`` protocol.
    start_dow
        Day-of-week on which the calendar-week cycle begins
        (0=Monday, 6=Sunday, ISO convention). Use ``6`` for MMWR
        Sunday-Saturday epiweeks or ``0`` for ISO Monday-Sunday weeks.
    """

    step_size = 7
    requires_calendar_anchor = True

    def __init__(self, inner: TemporalProcess, start_dow: int) -> None:
        """
        Initialize weekly temporal process.

        Parameters
        ----------
        inner
            Inner ``TemporalProcess`` that generates the weekly trajectory.
        start_dow
            Day-of-week on which the calendar-week cycle begins
            (0=Monday, 6=Sunday, ISO convention).
        """
        validate_dow(start_dow, "start_dow")
        self.inner = inner
        self.start_dow = start_dow

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"WeeklyTemporalProcess(inner={self.inner!r}, start_dow={self.start_dow!r})"
        )

    def _resolve_n_weekly(self, n_timepoints: int, first_day_dow: int | None) -> int:
        """
        Return the number of weekly samples needed.

        Returns
        -------
        int
            Number of weekly samples required to cover ``n_timepoints`` under
            the configured calendar-week cycle.
        """
        if first_day_dow is None:
            raise ValueError(
                "first_day_dow is required at sample time for WeeklyTemporalProcess"
            )
        validate_dow(first_day_dow, "first_day_dow")
        trim = (first_day_dow - self.start_dow) % 7
        return (n_timepoints + trim + 6) // 7

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int = 1,
        name_prefix: str = "weekly",
        *,
        first_day_dow: int | None = None,
    ) -> ArrayLike:
        """
        Sample weekly trajectory from inner process and broadcast.

        Computes the number of weekly time steps needed for the configured
        calendar-week cycle, samples the inner process at that cadence, then
        broadcasts each weekly value to the per-timepoint axis and trims to
        ``n_timepoints``. The returned value always has one row per evaluation
        timepoint, regardless of the inner parameter cadence. The weekly
        trajectory is recorded as a NumPyro deterministic site named
        ``"{name_prefix}_weekly"``.

        Parameters
        ----------
        n_timepoints
            Number of per-timepoint outputs to produce.
        initial_value
            Initial value(s) for the inner process. Defaults to 0.0.
        n_processes
            Number of parallel processes.
        name_prefix
            Prefix for numpyro sample sites; forwarded to the inner process.
        first_day_dow
            Day of week for element 0 of the shared model time axis
            (0=Monday, ..., 6=Sunday). Required.

        Returns
        -------
        ArrayLike
            Trajectories of shape ``(n_timepoints, n_processes)``, constant
            within each calendar week.
        """
        n_steps = self._resolve_n_weekly(n_timepoints, first_day_dow)
        # first_day_dow intentionally not forwarded: inner operates on the
        # weekly axis; the outer's axis-origin day-of-week does not apply.
        weekly = self.inner.sample(
            n_timepoints=n_steps,
            initial_value=initial_value,
            n_processes=n_processes,
            name_prefix=name_prefix,
        )
        numpyro.deterministic(f"{name_prefix}_weekly", weekly)
        return weekly_to_daily(
            weekly,
            week_start_dow=self.start_dow,
            output_data_first_dow=first_day_dow,
        )[:n_timepoints]
