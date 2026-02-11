"""
Multi-signal renewal model.

Combines a latent infection process with multiple observation processes.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.handlers

from pyrenew.latent.base import BaseLatentInfectionProcess
from pyrenew.metaclass import Model
from pyrenew.observation.base import BaseObservationProcess


class MultiSignalModel(Model):
    """
    Multi-signal renewal model.

    Combines a latent infection process (e.g., HierarchicalInfections,
    PartitionedInfections) with multiple observation processes (e.g.,
    CountObservation, WastewaterObservation).

    Built via PyrenewBuilder to ensure n_initialization_points is computed
    correctly from all components. Can also be constructed manually for
    advanced use cases.

    Parameters
    ----------
    latent_process : BaseLatentInfectionProcess
        Latent infection process generating infections at jurisdiction and/or
        subpopulation levels
    observations : Dict[str, BaseObservationProcess]
        Dictionary mapping names to observation process instances. Names are
        used when passing observation data to sample().

    Notes
    -----
    The model automatically routes latent infections to observations based on
    each observation's infection_resolution() method:
    - "aggregate" → receives aggregate infections from latent process
    - "subpop" → receives all subpopulation infections; observation selects via indices
    """

    def __init__(
        self,
        latent_process: BaseLatentInfectionProcess,
        observations: dict[str, BaseObservationProcess],
    ):
        """
        Initialize multi-signal model.

        Parameters
        ----------
        latent_process
            Configured latent infection process
        observations
            Dictionary mapping observation names to observation process instances

        Raises
        ------
        ValueError
            If validation fails (e.g., observation requires subpopulations
            but latent process doesn't support them)
        """
        self.latent = latent_process
        self.observations = observations
        self.validate()

    _SUPPORTED_RESOLUTIONS = {"aggregate", "subpop"}

    def validate(self) -> None:
        """
        Validate that observation processes are compatible with latent process.

        Checks that each observation implements infection_resolution()
        and returns a supported resolution.

        Raises
        ------
        ValueError
            If an observation doesn't implement infection_resolution()
            or returns an unsupported resolution.
        """
        for name, obs in self.observations.items():
            if not hasattr(obs, "infection_resolution"):
                raise ValueError(
                    f"Observation '{name}' must implement infection_resolution()"
                )
            resolution = obs.infection_resolution()
            if resolution not in self._SUPPORTED_RESOLUTIONS:
                raise ValueError(
                    f"Observation '{name}' returned invalid infection_resolution "
                    f"'{resolution}'. Expected one of {self._SUPPORTED_RESOLUTIONS}."
                )

    def pad_observations(
        self,
        obs: jnp.ndarray,
        axis: int = 0,
    ) -> jnp.ndarray:
        """
        Pad observations with NaN for the initialization period.

        Observation data uses a shared time axis [0, n_total) where
        n_total = n_init + n_days. This method prepends n_init NaN values
        to align user data (starting at day 0 of observations) with the
        shared axis.

        Parameters
        ----------
        obs : ArrayLike
            Observations in natural coordinates (index 0 = first observation day).
            Integer arrays are converted to float (required for NaN).
        axis : int, default 0
            Axis along which to pad (typically 0 for time axis).

        Returns
        -------
        jnp.ndarray
            Padded observations. First n_init values are NaN.
        """
        n_init = self.latent.n_initialization_points
        obs = jnp.asarray(obs, dtype=float)
        pad_shape = list(obs.shape)
        pad_shape[axis] = n_init
        padding = jnp.full(pad_shape, jnp.nan)
        return jnp.concatenate([padding, obs], axis=axis)

    def shift_times(self, times: jnp.ndarray) -> jnp.ndarray:
        """
        Shift time indices from natural coordinates to shared time axis.

        Observation data uses a shared time axis [0, n_total) where
        n_total = n_init + n_days. User-provided times in natural coordinates
        (0 = first observation day) must be shifted by n_init to align with
        the shared axis.

        Parameters
        ----------
        times : ArrayLike
            Time indices in natural coordinates (0 = first observation day).

        Returns
        -------
        jnp.ndarray
            Time indices on the shared axis [n_init, n_total).
        """
        n_init = self.latent.n_initialization_points
        return jnp.asarray(times) + n_init

    def validate_data(
        self,
        n_days_post_init: int,
        subpop_fractions=None,
        **observation_data: dict[str, Any],
    ) -> None:
        """
        Validate observation data before running MCMC.

        All observation data uses a shared time axis [0, n_total) where
        n_total = n_init + n_days_post_init. Dense observations must have
        length n_total with NaN padding for the initialization period.
        Sparse observations provide times indices on this shared axis.

        This method must be called with concrete (non-traced) values
        before running inference. Validation using Python control flow
        (if/raise) cannot be done during JAX tracing.

        Parameters
        ----------
        n_days_post_init : int
            Number of days to simulate after initialization period
        subpop_fractions : ArrayLike
            Population fractions for all subpopulations. Shape: (n_subpops,).
        **observation_data
            Data for each observation process, keyed by observation name.
            Each value should be a dict of kwargs for that observation's sample().

        Raises
        ------
        ValueError
            If times indices are out of bounds or negative
            If dense obs length doesn't match n_total
            If data shapes are inconsistent
        """
        pop = BaseLatentInfectionProcess._parse_and_validate_fractions(
            subpop_fractions=subpop_fractions,
        )

        n_init = self.latent.n_initialization_points
        n_total = n_init + n_days_post_init

        for name, obs_data in observation_data.items():
            if name not in self.observations:
                raise ValueError(
                    f"Unknown observation '{name}'. "
                    f"Available: {list(self.observations.keys())}"
                )

            self.observations[name].validate_data(
                n_total=n_total,
                n_subpops=pop.n_subpops,
                **obs_data,
            )

    def sample(
        self,
        n_days_post_init: int,
        population_size: float,
        *,
        subpop_fractions=None,
        **observation_data,
    ):
        """
        Sample from the joint generative model.

        This is the model function called by NumPyro during inference.

        Parameters
        ----------
        n_days_post_init : int
            Number of days to simulate after initialization period
        population_size : float
            Total population size. Used to convert infection proportions
            (from latent process) to infection counts (for observation processes).
        subpop_fractions : ArrayLike
            Population fractions for all subpopulations. Shape: (n_subpops,).
        **observation_data
            Data for each observation process, keyed by observation name
            (the ``name`` attribute of each observation process).
            Each value should be a dict of kwargs for that observation's sample().

        Returns
        -------
        None
            All quantities are recorded as NumPyro deterministic sites
            (``latent_infections``, ``latent_infections_by_subpop``) and
            observation sites. Use ``numpyro.infer.Predictive`` for forward
            sampling.
        """
        # Generate latent infections (proportions)
        latent_sample = self.latent.sample(
            n_days_post_init=n_days_post_init,
            subpop_fractions=subpop_fractions,
        )

        # Scale from proportions to counts
        inf_aggregate = latent_sample.aggregate * population_size
        inf_all = latent_sample.all_subpops * population_size

        # Record scaled infections for posterior analysis
        numpyro.deterministic("latent_infections", inf_aggregate)
        numpyro.deterministic("latent_infections_by_subpop", inf_all)

        # Map infection resolution to infection arrays
        latent_map = {
            "aggregate": inf_aggregate,
            "subpop": inf_all,
        }

        # Apply each observation process
        for name, obs_process in self.observations.items():
            # Get the appropriate latent infections based on observation type
            resolution = obs_process.infection_resolution()
            if resolution not in latent_map:
                raise ValueError(
                    f"Observation '{name}' returned invalid infection_resolution "
                    f"'{resolution}'. Expected one of {self._SUPPORTED_RESOLUTIONS}."
                )
            latent_infections = latent_map[resolution]

            # Get observation-specific data
            obs_data = observation_data.get(name, {})

            # Sample from observation process
            obs_process.sample(
                infections=latent_infections,
                **obs_data,
            )

        return None
