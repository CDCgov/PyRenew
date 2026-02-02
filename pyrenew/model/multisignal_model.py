"""
Multi-signal renewal model.

Combines a latent infection process with multiple observation processes.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.handlers
from numpyro.infer import MCMC, NUTS

from pyrenew.latent.base import BaseLatentInfectionProcess
from pyrenew.metaclass import Model
from pyrenew.observation.base import BaseObservationProcess


class MultiSignalModel(Model):
    """
    Multi-signal renewal model.

    Combines a latent infection process (e.g., HierarchicalInfections,
    PartitionedInfections) with multiple observation processes (e.g.,
    CountObservation, WastewaterObservation).

    Built via ModelBuilder to ensure n_initialization_points is computed
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
    - "subpop" → receives observed subpopulation infections from latent process
    """

    def __init__(
        self,
        latent_process: BaseLatentInfectionProcess,
        observations: Dict[str, BaseObservationProcess],
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
        self._validate()

    def _validate(self) -> None:
        """
        Validate that observation processes are compatible with latent process.

        Checks that each observation implements infection_resolution().

        Raises
        ------
        ValueError
            If an observation doesn't implement infection_resolution()
        """
        for name, obs in self.observations.items():
            try:
                obs.infection_resolution()
            except (NotImplementedError, AttributeError):
                raise ValueError(
                    f"Observation '{name}' must implement infection_resolution()"
                )

    def validate(self) -> None:
        """
        Validate the model configuration.

        This method is required by PyRenew's Model base class.
        Validation is performed in __init__ via _validate().

        Raises
        ------
        ValueError
            If validation fails
        """
        self._validate()

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
        obs_fractions=None,
        unobs_fractions=None,
        **observation_data: Dict[str, Any],
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
        obs_fractions : ArrayLike
            Population fractions for observed subpopulations
        unobs_fractions : ArrayLike
            Population fractions for unobserved subpopulations
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
            obs_fractions=obs_fractions,
            unobs_fractions=unobs_fractions,
        )

        n_init = self.latent.n_initialization_points
        n_total = n_init + n_days_post_init

        for name, obs_data in observation_data.items():
            if name not in self.observations:
                raise ValueError(
                    f"Unknown observation '{name}'. "
                    f"Available: {list(self.observations.keys())}"
                )

            obs = obs_data.get("obs")
            times = obs_data.get("times")

            if times is not None:
                # Sparse observations: times on shared axis [0, n_total)
                times = jnp.asarray(times)
                if jnp.any(times < 0):
                    raise ValueError(f"Observation '{name}': times cannot be negative")
                max_time = jnp.max(times)
                if max_time >= n_total:
                    raise ValueError(
                        f"Observation '{name}': times index {int(max_time)} "
                        f">= n_total ({n_total} = {n_init} init + "
                        f"{n_days_post_init} days). "
                        f"Times must be on shared axis [0, {n_total})."
                    )
                if obs is not None and len(obs) != len(times):
                    raise ValueError(
                        f"Observation '{name}': obs length {len(obs)} "
                        f"must match times length {len(times)}"
                    )
            elif obs is not None:
                # Dense observations: length must equal n_total
                obs = jnp.asarray(obs)
                if obs.shape[0] != n_total:
                    raise ValueError(
                        f"Observation '{name}': obs length {obs.shape[0]} "
                        f"must equal n_total ({n_total} = {n_init} init + "
                        f"{n_days_post_init} days). "
                        f"Pad with NaN for initialization period."
                    )

            # Validate subpop_indices if present
            subpop_indices = obs_data.get("subpop_indices")
            if subpop_indices is not None:
                subpop_indices = jnp.asarray(subpop_indices)
                if jnp.any(subpop_indices < 0):
                    raise ValueError(
                        f"Observation '{name}': subpop_indices cannot be negative"
                    )
                max_idx = jnp.max(subpop_indices)
                if max_idx >= pop.K_obs:
                    raise ValueError(
                        f"Observation '{name}': subpop_indices contains "
                        f"{int(max_idx)} >= {pop.K_obs} (K_obs)"
                    )

    def sample(
        self,
        n_days_post_init: int,
        population_size: float,
        *,
        obs_fractions=None,
        unobs_fractions=None,
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
        obs_fractions : ArrayLike
            Population fractions for observed subpopulations.
        unobs_fractions : ArrayLike
            Population fractions for unobserved subpopulations.
        **observation_data
            Data for each observation process, keyed by observation name
            (the ``name`` attribute of each observation process).
            Each value should be a dict of kwargs for that observation's sample().

        Returns
        -------
        tuple
            Four arrays of infection counts (scaled by population_size):
            - inf_aggregate: shape (n_total_days,) - aggregate infections
            - inf_all: shape (n_total_days, K) - all subpopulation infections
            - inf_obs: shape (n_total_days, K_obs) - observed subpopulation infections
            - inf_unobs: shape (n_total_days, K_unobs) - unobserved subpopulation infections

            where n_total_days = n_initialization_points + n_days_post_init
        """
        # Generate latent infections (proportions)
        latent_sample = self.latent.sample(
            n_days_post_init=n_days_post_init,
            obs_fractions=obs_fractions,
            unobs_fractions=unobs_fractions,
        )

        # Scale from proportions to counts
        inf_aggregate = latent_sample.aggregate * population_size
        inf_all = latent_sample.all_subpops * population_size
        inf_obs = latent_sample.observed * population_size
        inf_unobs = latent_sample.unobserved * population_size

        # Record scaled infections for posterior analysis
        numpyro.deterministic("latent_infections", inf_aggregate)
        numpyro.deterministic("latent_infections_by_subpop", inf_all)

        # Map infection resolution to infection arrays
        latent_map = {
            "aggregate": inf_aggregate,
            "subpop": inf_obs,
        }

        # Apply each observation process
        for name, obs_process in self.observations.items():
            # Get the appropriate latent infections based on observation type
            resolution = obs_process.infection_resolution()
            if resolution not in latent_map:
                raise ValueError(
                    f"Observation '{name}' returned invalid infection_resolution "
                    f"'{resolution}'. Expected 'aggregate' or 'subpop'."
                )
            latent_infections = latent_map[resolution]

            # Get observation-specific data
            obs_data = observation_data.get(name, {})

            # Sample from observation process
            obs_process.sample(
                infections=latent_infections,
                **obs_data,
            )

        # Return scaled infection counts
        return inf_aggregate, inf_all, inf_obs, inf_unobs

    def fit(
        self,
        n_days_post_init: int,
        population_size: float,
        *,
        obs_fractions=None,
        unobs_fractions=None,
        num_warmup: int = 500,
        num_samples: int = 500,
        num_chains: int = 1,
        rng_key: random.PRNGKey = None,
        reparam_config: Dict[str, Any] | None = None,
        progress_bar: bool = True,
        **observation_data,
    ):
        """
        Fit the model to observed data via MCMC.

        Validates observation data, runs NUTS sampler, and returns
        posterior samples.

        Parameters
        ----------
        n_days_post_init : int
            Number of days to simulate after initialization period
        population_size : float
            Total population size for scaling infections
        obs_fractions : ArrayLike
            Population fractions for observed subpopulations.
        unobs_fractions : ArrayLike
            Population fractions for unobserved subpopulations.
        num_warmup : int
            Number of MCMC warmup iterations (default: 500)
        num_samples : int
            Number of MCMC samples to draw (default: 500)
        num_chains : int
            Number of MCMC chains (default: 1)
        rng_key : jax.random.PRNGKey, optional
            Random key for reproducibility. Defaults to PRNGKey(0).
        reparam_config : dict, optional
            Reparameterization configuration for numpyro.handlers.reparam.
            Maps sample site names to reparameterizers (e.g., LocScaleReparam).
            Use non-centered parameterization (centered=0) for sparse data,
            centered (centered=1) for abundant data. Default None (no reparam).
        progress_bar : bool
            Whether to show progress bar during MCMC (default: True).
            Set to False if experiencing tqdm widget errors in Jupyter notebooks
            with parallel chains.
        **observation_data
            Data for each observation process, keyed by observation name.
            Each value should be a dict with 'counts', 'times', etc.

        Returns
        -------
        numpyro.infer.MCMC
            The MCMC object after sampling. Use `mcmc.get_samples()` to get
            posterior samples as a dict, or access other MCMC diagnostics.
        """
        self.validate_data(
            n_days_post_init=n_days_post_init,
            obs_fractions=obs_fractions,
            unobs_fractions=unobs_fractions,
            **observation_data,
        )

        if rng_key is None:
            rng_key = random.PRNGKey(0)

        # Apply reparameterization if config provided
        sample_fn = self.sample
        if reparam_config is not None:
            sample_fn = numpyro.handlers.reparam(self.sample, config=reparam_config)

        kernel = NUTS(sample_fn)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )
        mcmc.run(
            rng_key,
            n_days_post_init=n_days_post_init,
            population_size=population_size,
            obs_fractions=obs_fractions,
            unobs_fractions=unobs_fractions,
            **observation_data,
        )

        return mcmc
