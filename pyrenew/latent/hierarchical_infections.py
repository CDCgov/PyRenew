"""
Hierarchical latent infection process with subpopulation-specific renewal models.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike

from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent.base import (
    BaseLatentInfectionProcess,
    LatentSample,
)
from pyrenew.latent.infection_functions import compute_infections_from_rt
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.math import r_approx_from_R
from pyrenew.metaclass import RandomVariable


class HierarchicalInfections(BaseLatentInfectionProcess):
    """
    Multi-subpopulation renewal model with hierarchical Rt structure.

    Each subpopulation has its own renewal equation with Rt deviating from a
    shared baseline. Suitable when transmission dynamics vary substantially
    across subpopulations.

    Mathematical form:
    - Baseline Rt: log[R_baseline(t)] ~ TemporalProcess
    - Subpopulation Rt: log R_k(t) = log[R_baseline(t)] + delta_k(t)
    - Deviations: delta_k(t) ~ TemporalProcess with sum-to-zero constraint
    - Renewal per subpop: I_k(t) = R_k(t) * sum_tau I_k(t-tau) * g(tau)
    - Aggregate total: I_aggregate(t) = sum_k p_k * I_k(t)

    The constructor specifies model structure (priors, temporal processes).
    Population structure (fractions, K, K_obs) is provided at sample time,
    allowing a single model to be fit to multiple jurisdictions.

    Parameters
    ----------
    gen_int_rv : RandomVariable
        Generation interval PMF
    I0_rv : RandomVariable
        Initial infection prevalence (proportion of population) at first
        observation time. Must return values in the interval (0, 1).
        Returns scalar (same for all subpops) or (K,) array (per-subpop).
        Full I0 matrix generated via exponential backprojection during sampling.
    baseline_temporal : TemporalProcess
        Temporal process for baseline Rt dynamics
    subpop_temporal : TemporalProcess
        Temporal process for subpopulation deviations
    initial_log_rt_rv : RandomVariable
        Initial value for log(Rt) at time 0.  Can be estimated from data
        or given a prior distribution.
    n_initialization_points : int, optional
        Number of initialization days before day 0. If not specified, defaults to
        max(21, 2*len(gen_int)-1). When using ModelBuilder, this is computed
        automatically from all observation processes.

    Notes
    -----
    Sum-to-zero constraint on deviations ensures R_baseline(t) is the geometric
    mean of subpopulation Rt values, providing identifiability.

    When using ModelBuilder (recommended), n_initialization_points is computed
    automatically from all observation processes and should not be specified manually.
    """

    def __init__(
        self,
        *,
        gen_int_rv: RandomVariable,
        I0_rv: RandomVariable,
        baseline_temporal: TemporalProcess,
        subpop_temporal: TemporalProcess,
        initial_log_rt_rv: RandomVariable,
        n_initialization_points: int = None,
    ) -> None:
        """
        Initialize hierarchical infections process.

        Parameters
        ----------
        gen_int_rv : RandomVariable
            Generation interval PMF
        I0_rv : RandomVariable
            Initial infection prevalence (proportion of population)
        baseline_temporal : TemporalProcess
            Temporal process for baseline Rt dynamics
        subpop_temporal : TemporalProcess
            Temporal process for subpopulation deviations
        initial_log_rt_rv : RandomVariable
            Initial value for log(Rt) at time 0.
        n_initialization_points : int, optional
            Number of initialization days before day 0.

        Raises
        ------
        ValueError
            If required parameters are missing or invalid
        """
        super().__init__(
            gen_int_rv=gen_int_rv,
            n_initialization_points=n_initialization_points,
        )

        if I0_rv is None:
            raise ValueError("I0_rv is required")
        self.I0_rv = I0_rv

        # Validate I0 at construction time if it's deterministic
        if isinstance(I0_rv, DeterministicVariable):
            self._validate_I0(I0_rv.value)

        # Validate initial log Rt
        if initial_log_rt_rv is None:
            raise ValueError("initial_log_rt_rv is required")
        self.initial_log_rt_rv = initial_log_rt_rv

        if baseline_temporal is None:
            raise ValueError("baseline_temporal is required")
        self.baseline_temporal = baseline_temporal

        if subpop_temporal is None:
            raise ValueError("subpop_temporal is required")
        self.subpop_temporal = subpop_temporal

    def validate(self) -> None:
        """
        Validate hierarchical infections parameters.

        Checks that temporal processes and I0_rv can be sampled.

        Raises
        ------
        ValueError
            If any parameters fail validation
        """
        gen_int_pmf = self.gen_int_rv()
        if gen_int_pmf.size == 0:  # pragma: no cover
            raise ValueError("gen_int_rv must return non-empty array")

        pmf_sum = jnp.sum(gen_int_pmf)
        if not jnp.isclose(pmf_sum, 1.0, atol=1e-6):  # pragma: no cover
            raise ValueError(f"gen_int_rv must sum to 1.0, got {float(pmf_sum):.6f}")

        if jnp.any(gen_int_pmf < 0):  # pragma: no cover
            raise ValueError("gen_int_rv must have non-negative values")

    def sample(
        self,
        n_days_post_init: int,
        *,
        obs_fractions: ArrayLike = None,
        unobs_fractions: ArrayLike = None,
        **kwargs,
    ) -> LatentSample:
        """
        Sample hierarchical infections for all subpopulations.

        Generates baseline Rt, subpopulation deviations with sum-to-zero
        constraint, initial infections, and runs K independent renewal processes.

        Parameters
        ----------
        n_days_post_init : int
            Number of days to simulate after initialization period
        obs_fractions : ArrayLike
            Population fractions for observed subpopulations.
        unobs_fractions : ArrayLike
            Population fractions for unobserved subpopulations.
        **kwargs
            Additional arguments (unused, for compatibility)

        Returns
        -------
        LatentSample
            Named tuple with fields:
            - aggregate: shape (n_total_days,)
            - all_subpops: shape (n_total_days, K)
            - observed: shape (n_total_days, K_obs)
            - unobserved: shape (n_total_days, K_unobs)
        """
        # Parse and validate population structure
        pop = self._parse_and_validate_fractions(
            obs_fractions=obs_fractions,
            unobs_fractions=unobs_fractions,
        )

        n_total_days = self.n_initialization_points + n_days_post_init

        initial_log_rt = self.initial_log_rt_rv()

        log_rt_baseline = self.baseline_temporal.sample(
            n_timepoints=n_total_days,
            initial_value=initial_log_rt,
            name_prefix="log_rt_baseline",
        )

        deviations_raw = self.subpop_temporal.sample(
            n_timepoints=n_total_days,
            n_processes=pop.K,
            initial_value=jnp.zeros(pop.K),
            name_prefix="subpop_deviations",
        )

        # Sum-to-zero constraint ensures identifiability
        mean_deviation = jnp.mean(deviations_raw, axis=1, keepdims=True)
        deviations = deviations_raw - mean_deviation

        log_rt_subpop = log_rt_baseline[:, jnp.newaxis] + deviations
        rt_subpop = jnp.exp(log_rt_subpop)

        gen_int = self.gen_int_rv()

        I0 = jnp.asarray(self.I0_rv())
        self._validate_I0(I0)

        if I0.ndim == 0:
            I0_subpop = jnp.full(pop.K, I0)
        else:
            I0_subpop = I0

        initial_r_subpop = jax.vmap(
            partial(r_approx_from_R, g=gen_int, n_newton_steps=4)
        )(rt_subpop[0, :])

        # Vectorized exponential growth initialization for all subpopulations
        # Formula: I0_subpop[k] * exp(initial_r_subpop[k] * t) for t in [0, n_init)
        time_indices = jnp.arange(self.n_initialization_points)
        I0_all = I0_subpop[jnp.newaxis, :] * jnp.exp(
            initial_r_subpop[jnp.newaxis, :] * time_indices[:, jnp.newaxis]
        )
        numpyro.deterministic("I0_init_all_subpops", I0_all)

        gen_int_reversed = jnp.flip(gen_int)
        recent_I0_all = I0_all[-gen_int.size :, :]

        all_fractions = jnp.concatenate([pop.obs_fractions, pop.unobs_fractions])

        # Vectorized renewal equation for all subpopulations via vmap
        post_init_infections_all = jax.vmap(
            lambda I0_col, Rt_col: compute_infections_from_rt(
                I0=I0_col,
                Rt=Rt_col,
                reversed_generation_interval_pmf=gen_int_reversed,
            ),
            in_axes=1,
            out_axes=1,
        )(recent_I0_all, rt_subpop[self.n_initialization_points :, :])

        infections_all = jnp.vstack([I0_all, post_init_infections_all])

        infections_aggregate = jnp.sum(
            infections_all * all_fractions[jnp.newaxis, :], axis=1
        )

        infections_observed, infections_unobserved = self._split_subpopulations(
            infections_all, pop
        )

        self._validate_output_shapes(
            infections_aggregate,
            infections_all,
            infections_observed,
            infections_unobserved,
            n_total_days,
            pop,
        )

        # Record key quantities for diagnostics and posterior analysis
        numpyro.deterministic("log_rt_baseline", log_rt_baseline)
        numpyro.deterministic("rt_baseline", jnp.exp(log_rt_baseline))
        numpyro.deterministic("rt_subpop", rt_subpop)
        numpyro.deterministic("subpop_deviations", deviations)
        numpyro.deterministic("infections_aggregate", infections_aggregate)

        return LatentSample(
            aggregate=infections_aggregate,
            all_subpops=infections_all,
            observed=infections_observed,
            unobserved=infections_unobserved,
        )
