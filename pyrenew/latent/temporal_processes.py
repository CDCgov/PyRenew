# numpydoc ignore=GL08
"""
Temporal processes for latent infection models.

Provides time-series processes for modeling Rt dynamics and subpopulation
deviations in hierarchical infection models. All processes support both
single trajectory and vectorized (multiple parallel trajectories) sampling
through a unified ``TemporalProcess`` protocol.

Relationship to pyrenew.process
-------------------------------
This module provides **high-level, domain-specific wrappers** around the
low-level building blocks in :mod:`pyrenew.process`. The key differences:

.. list-table::
   :header-rows: 1

   * - Aspect
     - ``pyrenew.process``
     - ``pyrenew.latent.temporal_processes``
   * - Abstraction level
     - Low-level composable primitives
     - High-level domain-specific API
   * - Interface
     - Varied signatures per class
     - Unified ``TemporalProcess`` protocol
   * - Target use
     - General time-series modeling
     - Rt dynamics, hierarchical infections
   * - Vectorization
     - Caller manages array shapes
     - Automatic via ``n_processes`` parameter
   * - Validation
     - Minimal constraints
     - Validates positive innovation_sd

**When to use which:**

- Use ``pyrenew.process`` classes (``ARProcess``, ``DifferencedProcess``,
  ``RandomWalk``) when building novel statistical models or when you need
  fine-grained control over array shapes and numpyro sampling semantics.

- Use this module's classes (``AR1``, ``DifferencedAR1``, ``RandomWalk``)
  when modeling Rt trajectories in hierarchical infection models. These
  provide a consistent interface, automatic vectorization, and enforce
  epidemiologically-sensible constraints.

Temporal Processes
------------------
- ``AR1``: Autoregressive process with mean reversion. Keeps Rt bounded
  near a baseline. Wraps :class:`pyrenew.process.ARProcess`.
- ``DifferencedAR1``: AR(1) on first differences. Allows persistent trends
  while stabilizing the growth rate. Wraps :class:`pyrenew.process.DifferencedProcess`.
- ``RandomWalk``: No mean reversion. Rt can drift without bound.
  Wraps :class:`pyrenew.process.RandomWalk`.

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
from pyrenew.process import RandomWalk as PyRenewRandomWalk
from pyrenew.randomvariable import DistributionalVariable


@runtime_checkable
class TemporalProcess(Protocol):
    """
    Protocol for temporal processes generating time-varying parameters.

    Used for jurisdiction-level Rt dynamics, subpopulation deviations, or
    allocation trajectories. Supports both single trajectory and vectorized
    (multiple parallel trajectories) sampling through a unified interface.
    """

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int | None = None,
        name_prefix: str = "temporal",
    ) -> ArrayLike:
        """
        Sample temporal trajectory or trajectories.

        Parameters
        ----------
        n_timepoints : int
            Number of time points to generate
        initial_value : float or ArrayLike, optional
            Initial value(s) for the process(es).
            - If n_processes is None: scalar initial value (default: 0.0)
            - If n_processes is K: scalar (broadcast) or array of shape (K,)
        n_processes : int, optional
            Number of parallel processes. If None, samples a single trajectory.
        name_prefix : str, default "temporal"
            Prefix for numpyro sample site names to avoid collisions

        Returns
        -------
        ArrayLike
            - If n_processes is None: trajectory of shape (n_timepoints,)
            - If n_processes is K: trajectories of shape (n_timepoints, K)
        """
        ...


class AR1(TemporalProcess):
    """
    AR(1) process.

    Each value depends on the previous value plus noise, with reversion
    toward a mean level. Keeps Rt bounded near a baseline — values that
    drift away are "pulled back" over time.

    This class wraps :class:`pyrenew.process.ARProcess` with a simplified,
    protocol-compliant interface that handles vectorization automatically.

    Parameters
    ----------
    autoreg : float
        Autoregressive coefficient. For stationarity, |autoreg| < 1, but
        this is not enforced (use priors to constrain if needed).
    innovation_sd : float, default 1.0
        Standard deviation of noise at each time step. Larger values produce
        more volatile trajectories; smaller values produce smoother ones.
    """

    def __init__(self, autoreg: float, innovation_sd: float = 1.0):
        """
        Initialize AR(1) process.

        Parameters
        ----------
        autoreg : float
            Autoregressive coefficient. For stationarity, |autoreg| < 1,
            but this is not enforced (use priors to constrain if needed).
        innovation_sd : float, default 1.0
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
        self.ar_process = ARProcess()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AR1(autoreg={self.autoreg}, innovation_sd={self.innovation_sd})"

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int | None = None,
        name_prefix: str = "ar1",
    ) -> ArrayLike:
        """
        Sample AR(1) trajectory or trajectories.

        Parameters
        ----------
        n_timepoints : int
            Number of time points to generate
        initial_value : float or ArrayLike, optional
            Initial value(s). Defaults to 0.0.
        n_processes : int, optional
            Number of parallel processes. If None, samples single trajectory.
        name_prefix : str, default "ar1"
            Prefix for numpyro sample sites

        Returns
        -------
        ArrayLike
            Trajectory (n_timepoints,) or trajectories (n_timepoints, K)
        """
        if n_processes is None:
            return self._sample_single(n_timepoints, initial_value, name_prefix)
        else:
            return self._sample_vectorized(
                n_timepoints, n_processes, initial_value, name_prefix
            )

    def _sample_single(
        self,
        n_timepoints: int,
        initial_value: float | None,
        name_prefix: str,
    ) -> ArrayLike:
        """
        Sample a single AR(1) trajectory.

        Returns
        -------
        ArrayLike
            Trajectory of shape (n_timepoints,).
        """
        if initial_value is None:
            initial_value = 0.0

        stationary_sd = self.innovation_sd / jnp.sqrt(1 - self.autoreg**2)
        init_state = numpyro.sample(
            f"{name_prefix}_init", dist.Normal(initial_value, stationary_sd)
        )

        trajectory = self.ar_process(
            n=n_timepoints,
            init_vals=jnp.array([init_state]),
            autoreg=jnp.array([self.autoreg]),
            noise_sd=self.innovation_sd,
            noise_name=f"{name_prefix}_noise",
        )

        return trajectory

    def _sample_vectorized(
        self,
        n_timepoints: int,
        n_processes: int,
        initial_values: ArrayLike | None,
        name_prefix: str,
    ) -> ArrayLike:
        """
        Sample multiple AR(1) trajectories in parallel.

        Returns
        -------
        ArrayLike
            Trajectories of shape (n_timepoints, n_processes).
        """
        if initial_values is None:
            initial_values = jnp.zeros(n_processes)
        elif jnp.isscalar(initial_values):
            initial_values = jnp.full(n_processes, initial_values)

        stationary_sd = self.innovation_sd / jnp.sqrt(1 - self.autoreg**2)

        with numpyro.plate(f"{name_prefix}_init_plate", n_processes):
            init_states = numpyro.sample(
                f"{name_prefix}_init",
                dist.Normal(initial_values, stationary_sd),
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

    This class wraps :class:`pyrenew.process.DifferencedProcess` with
    :class:`pyrenew.process.ARProcess` as the fundamental process, providing
    a simplified, protocol-compliant interface.

    Parameters
    ----------
    autoreg : float
        Autoregressive coefficient for differences. For stationarity,
        |autoreg| < 1, but this is not enforced (use priors to constrain
        if needed).
    innovation_sd : float, default 1.0
        Standard deviation of noise added to changes. Larger values produce
        more erratic growth rates; smaller values produce smoother trends.
    """

    def __init__(self, autoreg: float, innovation_sd: float = 1.0):
        """
        Initialize differenced AR(1) process.

        Parameters
        ----------
        autoreg : float
            Autoregressive coefficient for differences. For stationarity,
            |autoreg| < 1, but this is not enforced (use priors to constrain
            if needed).
        innovation_sd : float, default 1.0
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
            fundamental_process=ARProcess(), differencing_order=1
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DifferencedAR1(autoreg={self.autoreg}, innovation_sd={self.innovation_sd})"

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int | None = None,
        name_prefix: str = "diff_ar1",
    ) -> ArrayLike:
        """
        Sample differenced AR(1) trajectory or trajectories.

        Parameters
        ----------
        n_timepoints : int
            Number of time points to generate
        initial_value : float or ArrayLike, optional
            Initial value(s). Defaults to 0.0.
        n_processes : int, optional
            Number of parallel processes. If None, samples single trajectory.
        name_prefix : str, default "diff_ar1"
            Prefix for numpyro sample sites

        Returns
        -------
        ArrayLike
            Trajectory (n_timepoints,) or trajectories (n_timepoints, K)
        """
        if n_processes is None:
            return self._sample_single(n_timepoints, initial_value, name_prefix)
        else:
            return self._sample_vectorized(
                n_timepoints, n_processes, initial_value, name_prefix
            )

    def _sample_single(
        self,
        n_timepoints: int,
        initial_value: float | None,
        name_prefix: str,
    ) -> ArrayLike:
        """
        Sample a single differenced AR(1) trajectory.

        Returns
        -------
        ArrayLike
            Trajectory of shape (n_timepoints,).
        """
        if initial_value is None:
            initial_value = 0.0

        stationary_sd = self.innovation_sd / jnp.sqrt(1 - self.autoreg**2)
        init_rate_of_change = numpyro.sample(
            f"{name_prefix}_init_rate", dist.Normal(0, stationary_sd)
        )

        trajectory = self.process(
            n=n_timepoints + 1,
            init_vals=jnp.array([initial_value]),
            autoreg=jnp.array([self.autoreg]),
            noise_sd=self.innovation_sd,
            fundamental_process_init_vals=jnp.array([init_rate_of_change]),
            noise_name=f"{name_prefix}_noise",
        )

        return trajectory[:n_timepoints]

    def _sample_vectorized(
        self,
        n_timepoints: int,
        n_processes: int,
        initial_values: ArrayLike | None,
        name_prefix: str,
    ) -> ArrayLike:
        """
        Sample multiple differenced AR(1) trajectories in parallel.

        Returns
        -------
        ArrayLike
            Trajectories of shape (n_timepoints, n_processes).
        """
        if initial_values is None:
            initial_values = jnp.zeros(n_processes)
        elif jnp.isscalar(initial_values):
            initial_values = jnp.full(n_processes, initial_values)

        stationary_sd = self.innovation_sd / jnp.sqrt(1 - self.autoreg**2)

        with numpyro.plate(f"{name_prefix}_init_rate_plate", n_processes):
            init_rates = numpyro.sample(
                f"{name_prefix}_init_rate",
                dist.Normal(0, stationary_sd),
            )

        trajectories = self.process(
            n=n_timepoints + 1,
            init_vals=initial_values[jnp.newaxis, :],
            autoreg=jnp.full((1, n_processes), self.autoreg),
            noise_sd=self.innovation_sd,
            fundamental_process_init_vals=init_rates[jnp.newaxis, :],
            noise_name=f"{name_prefix}_noise",
        )

        return trajectories[:n_timepoints, :]


class RandomWalk(TemporalProcess):
    """
    Random walk process for log(Rt).

    Each value equals the previous value plus noise, with no reversion
    toward a mean. Allows Rt to drift without bound — suitable when you
    have no prior expectation that Rt will return to a baseline.

    This class wraps :class:`pyrenew.process.RandomWalk` with a simplified,
    protocol-compliant interface. The vectorized mode uses a non-centered
    parameterization to avoid funnel problems in inference.

    Parameters
    ----------
    innovation_sd : float, default 1.0
        Standard deviation of noise at each time step. Larger values produce
        faster drift; smaller values produce more gradual changes.

    Notes
    -----
    Unlike AR(1), variance grows over time — the process can wander arbitrarily
    far from its starting point. For long time horizons, consider AR(1) if you
    want Rt to stay bounded near a baseline.
    """

    def __init__(self, innovation_sd: float = 1.0):
        """
        Initialize random walk process.

        Parameters
        ----------
        innovation_sd : float, default 1.0
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
        n_processes: int | None = None,
        name_prefix: str = "rw",
    ) -> ArrayLike:
        """
        Sample random walk trajectory or trajectories.

        Parameters
        ----------
        n_timepoints : int
            Number of time points to generate
        initial_value : float or ArrayLike, optional
            Initial value(s). Defaults to 0.0.
        n_processes : int, optional
            Number of parallel processes. If None, samples single trajectory.
        name_prefix : str, default "rw"
            Prefix for numpyro sample sites

        Returns
        -------
        ArrayLike
            Trajectory (n_timepoints,) or trajectories (n_timepoints, K)
        """
        if n_processes is None:
            return self._sample_single(n_timepoints, initial_value, name_prefix)
        else:
            return self._sample_vectorized(
                n_timepoints, n_processes, initial_value, name_prefix
            )

    def _sample_single(
        self,
        n_timepoints: int,
        initial_value: float | None,
        name_prefix: str,
    ) -> ArrayLike:
        """
        Sample a single random walk trajectory using PyRenew's RandomWalk.

        Returns
        -------
        ArrayLike
            Trajectory of shape (n_timepoints,).
        """
        if initial_value is None:
            initial_value = 0.0

        step_rv = DistributionalVariable(
            name=f"{name_prefix}_step",
            distribution=dist.Normal(0, self.innovation_sd),
        )

        rw = PyRenewRandomWalk(step_rv=step_rv)

        trajectory = rw.sample(
            init_vals=jnp.array([initial_value]),
            n=n_timepoints,
        )

        return trajectory

    def _sample_vectorized(
        self,
        n_timepoints: int,
        n_processes: int,
        initial_values: ArrayLike | None,
        name_prefix: str,
    ) -> ArrayLike:
        """
        Sample multiple random walk trajectories in parallel.

        Returns
        -------
        ArrayLike
            Trajectories of shape (n_timepoints, n_processes).
        """
        if initial_values is None:
            initial_values = jnp.zeros(n_processes)
        elif jnp.isscalar(initial_values):
            initial_values = jnp.full(n_processes, initial_values)

        # Non-centered parameterization to avoid funnel problems
        with numpyro.plate(f"{name_prefix}_time", n_timepoints - 1):
            with numpyro.plate(f"{name_prefix}_proc", n_processes):
                increments_raw = numpyro.sample(
                    f"{name_prefix}_increments_raw",
                    dist.Normal(0, 1),
                )

        # Transpose: (n_processes, n_timepoints-1) -> (n_timepoints-1, n_processes)
        increments = numpyro.deterministic(
            f"{name_prefix}_increments",
            (increments_raw * self.innovation_sd).T,
        )

        cumulative = jnp.cumsum(increments, axis=0)

        return jnp.concatenate(
            [initial_values[jnp.newaxis, :], initial_values + cumulative],
            axis=0,
        )
