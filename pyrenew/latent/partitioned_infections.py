"""
Partitioned latent infection process with single renewal model and allocation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent.infection_functions import compute_infections_from_rt
from pyrenew.latent.infection_initialization_method import (
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.latent.infection_initialization_process import (
    InfectionInitializationProcess,
)
from pyrenew.math import r_approx_from_R
from pyrenew.metaclass import RandomVariable

from pyrenew.latent.base import (
    BaseLatentInfectionProcess,
    LatentSample,
)
from pyrenew.latent.temporal_processes import TemporalProcess


class PartitionedInfections(BaseLatentInfectionProcess):
    """
    Single aggregate renewal model with spatial allocation.

    Computes one renewal equation for total infections, then allocates to
    subpopulations via time-varying distribution. More efficient when spatial
    variation is in infection distribution rather than transmission dynamics.

    Mathematical form:
    - Aggregate Rt: log[R_aggregate(t)] ~ TemporalProcess
    - Single renewal: I_total(t) = R_aggregate(t) * sum_tau I_total(t-tau) * g(tau)
    - Baseline allocation: pi_k^base = population fractions
    - Allocation deviations: delta_k(t) ~ TemporalProcess, delta_1(t) = 0 (reference)
    - Softmax allocation: pi_k(t) = (pi_k^base * exp(delta_k)) / sum_j(...)
    - Subpop infections: I_k(t) = pi_k(t) * I_total(t)

    The constructor specifies model structure (priors, temporal processes).
    Population structure (fractions, K, K_obs) is provided at sample time,
    allowing a single model to be fit to multiple jurisdictions.

    Parameters
    ----------
    gen_int_rv : RandomVariable
        Generation interval PMF
    I0_rv : RandomVariable
        Initial infection prevalence (proportion of population) at first
        observation time. Must return value in the interval (0, 1].
        Returns scalar. Full I0 vector generated via exponential backprojection during sampling.
    initial_log_rt_rv : RandomVariable
        Initial value for log(Rt) at time 0.  Can be estimated from data
        or given a prior distribution.
    rt_temporal : TemporalProcess
        Temporal process for aggregate-level Rt
    allocation_temporal : TemporalProcess
        Temporal process for allocation deviations (K-1 processes, reference fixed to 0)
    n_initialization_points : int, optional
        Number of initialization days before day 0. If not specified, defaults to
        max(21, 2*len(gen_int)-1). When using ModelBuilder, this is computed
        automatically from all observation processes.

    Notes
    -----
    Reference category (subpopulation 0) has delta_0(t) = 0 for identifiability.
    Choose the largest or most representative subpopulation as index 0.
    """

    def __init__(
        self,
        *,
        gen_int_rv: RandomVariable,
        I0_rv: RandomVariable,
        rt_temporal: TemporalProcess,
        allocation_temporal: TemporalProcess,
        initial_log_rt_rv: RandomVariable,
        n_initialization_points: int = None,
    ) -> None:
        """
        Initialize partitioned infections process.

        Parameters
        ----------
        gen_int_rv : RandomVariable
            Generation interval PMF
        I0_rv : RandomVariable
            Initial infection prevalence (proportion of population)
        rt_temporal : TemporalProcess
            Temporal process for aggregate-level Rt
        allocation_temporal : TemporalProcess
            Temporal process for allocation deviations
        initial_log_rt_rv : RandomVariable
            Initial value for log(Rt) at time 0
        n_initialization_points : int, optional
            Number of initialization days before day 0

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

        if initial_log_rt_rv is None:
            raise ValueError("initial_log_rt_rv is required")
        self.initial_log_rt_rv = initial_log_rt_rv

        if rt_temporal is None:
            raise ValueError("rt_temporal is required")
        self.rt_temporal = rt_temporal

        if allocation_temporal is None:
            raise ValueError("allocation_temporal is required")
        self.allocation_temporal = allocation_temporal

    def validate(self) -> None:
        """
        Validate partitioned infections parameters.

        Checks that temporal processes and I0_rv can be sampled.

        Raises
        ------
        ValueError
            If any parameters fail validation
        """
        gen_int_pmf = self.gen_int_rv()
        if gen_int_pmf.size == 0:
            raise ValueError("gen_int_rv must return non-empty array")

        pmf_sum = jnp.sum(gen_int_pmf)
        if not jnp.isclose(pmf_sum, 1.0, atol=1e-6):
            raise ValueError(f"gen_int_rv must sum to 1.0, got {float(pmf_sum):.6f}")

        if jnp.any(gen_int_pmf < 0):
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
        Sample partitioned infections with single renewal model and allocation.

        Generates aggregate-level Rt, runs one renewal equation for total
        infections, then allocates to subpopulations via time-varying softmax.

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
            - aggregate: shape (n_total_days,) - total infections
            - all_subpops: shape (n_total_days, K) - allocated to all subpops
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

        log_rt_aggregate = self.rt_temporal.sample(
            n_timepoints=n_total_days,
            initial_value=initial_log_rt,
            innovation_sd=1.0,
            name_prefix="log_rt_aggregate",
        )
        rt_aggregate = jnp.exp(log_rt_aggregate)

        # Reference category (subpop 0) fixed to delta_0(t) = 0 for identifiability
        if pop.K > 1:
            deviations_nonref = self.allocation_temporal.sample(
                n_timepoints=n_total_days,
                n_processes=pop.K - 1,
                initial_value=jnp.zeros(pop.K - 1),
                innovation_sd=1.0,
                name_prefix="allocation_deviations",
            )
            deviations = jnp.concatenate(
                [jnp.zeros((n_total_days, 1)), deviations_nonref], axis=1
            )
        else:
            deviations = jnp.zeros((n_total_days, 1))

        baseline_allocation = jnp.concatenate([pop.obs_fractions, pop.unobs_fractions])

        # Softmax: pi_k(t) = (baseline_k * exp(delta_k(t))) / sum_j(baseline_j * exp(delta_j(t)))
        unnormalized = baseline_allocation[jnp.newaxis, :] * jnp.exp(deviations)
        allocation_proportions = unnormalized / jnp.sum(
            unnormalized, axis=1, keepdims=True
        )

        gen_int = self.gen_int_rv()
        I0_total_raw = self.I0_rv()
        I0_total = jnp.asarray(I0_total_raw)

        # Validate I0 at runtime (for stochastic RVs)
        self._validate_I0(I0_total)

        initial_r = r_approx_from_R(rt_aggregate[0], gen_int, n_newton_steps=4)

        i0_rv = DeterministicVariable("i0_prevalence", I0_total)
        r_rv = DeterministicVariable("initial_r", initial_r)

        init_proc = InfectionInitializationProcess(
            "I0_initialization",
            i0_rv,
            InitializeInfectionsExponentialGrowth(
                self.n_initialization_points,
                r_rv,
                t_pre_init=0,
            ),
        )

        I0_total = init_proc()

        gen_int_reversed = jnp.flip(gen_int)
        recent_I0_total = I0_total[-gen_int.size :]

        post_init_infections_total = compute_infections_from_rt(
            I0=recent_I0_total,
            Rt=rt_aggregate[self.n_initialization_points :],
            reversed_generation_interval_pmf=gen_int_reversed,
        )

        infections_total = jnp.concatenate([I0_total, post_init_infections_total])

        infections_all = allocation_proportions * infections_total[:, jnp.newaxis]

        infections_aggregate = infections_total

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
        numpyro.deterministic("log_rt_aggregate", log_rt_aggregate)
        numpyro.deterministic("rt_aggregate", rt_aggregate)
        numpyro.deterministic("allocation_deviations", deviations)
        numpyro.deterministic("allocation_proportions", allocation_proportions)
        numpyro.deterministic("infections_aggregate", infections_aggregate)

        return LatentSample(
            aggregate=infections_aggregate,
            all_subpops=infections_all,
            observed=infections_observed,
            unobserved=infections_unobserved,
        )
