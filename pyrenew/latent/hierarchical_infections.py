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
from pyrenew.distutil import validate_discrete_dist_vector
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
    Population structure (subpop_fractions) is provided at sample time,
    allowing a single model to be fit to multiple jurisdictions.

    Parameters
    ----------
    gen_int_rv : RandomVariable
        Generation interval PMF
    I0_rv : RandomVariable
        Initial infection prevalence (proportion of population) at first
        observation time. Must return values in the interval (0, 1).
        Returns scalar (same for all subpops) or (n_subpops,) array (per-subpop).
        Full I0 matrix generated via exponential backprojection during sampling.
    baseline_rt_process : TemporalProcess
        Temporal process for baseline Rt dynamics
    subpop_rt_deviation_process : TemporalProcess
        Temporal process for subpopulation deviations
    initial_log_rt_rv : RandomVariable
        Initial value for log(Rt) at time 0.  Can be estimated from data
        or given a prior distribution.
    n_initialization_points : int
        Number of initialization days before day 0. Must be at least
        ``len(gen_int_rv())`` to provide enough history for the renewal
        equation convolution. When using PyrenewBuilder, this is computed
        automatically from all observation processes.

    Notes
    -----
    Sum-to-zero constraint on deviations ensures R_baseline(t) is the geometric
    mean of subpopulation Rt values, providing identifiability.

    When using PyrenewBuilder (recommended), n_initialization_points is computed
    automatically from all observation processes.
    """

    def __init__(
        self,
        *,
        gen_int_rv: RandomVariable,
        I0_rv: RandomVariable,
        baseline_rt_process: TemporalProcess,
        subpop_rt_deviation_process: TemporalProcess,
        initial_log_rt_rv: RandomVariable,
        n_initialization_points: int,
        name: str = "latent_infections",
    ) -> None:
        """
        Initialize hierarchical infections process.

        Parameters
        ----------
        gen_int_rv : RandomVariable
            Generation interval PMF
        I0_rv : RandomVariable
            Initial infection prevalence (proportion of population)
        baseline_rt_process : TemporalProcess
            Temporal process for baseline Rt dynamics
        subpop_rt_deviation_process : TemporalProcess
            Temporal process for subpopulation deviations
        initial_log_rt_rv : RandomVariable
            Initial value for log(Rt) at time 0.
        n_initialization_points : int
            Number of initialization days before day 0.
        name : str
            Name prefix for numpyro sample sites. All deterministic
            quantities are recorded under this scope (e.g.,
            ``"{name}/rt_baseline"``). Default: ``"latent_infections"``.

        Raises
        ------
        ValueError
            If required parameters are missing or invalid
        """
        super().__init__(
            gen_int_rv=gen_int_rv,
            n_initialization_points=n_initialization_points,
        )
        self.name = name

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

        if baseline_rt_process is None:
            raise ValueError("baseline_rt_process is required")
        self.baseline_rt_process = baseline_rt_process

        if subpop_rt_deviation_process is None:
            raise ValueError("subpop_rt_deviation_process is required")
        self.subpop_rt_deviation_process = subpop_rt_deviation_process

    def validate(self) -> None:
        """
        Validate hierarchical infections parameters.

        Checks that the generation interval is a valid PMF.

        Raises
        ------
        ValueError
            If gen_int_rv does not return a valid discrete distribution
        """
        validate_discrete_dist_vector(self.gen_int_rv())

    def sample(
        self,
        n_days_post_init: int,
        *,
        subpop_fractions: ArrayLike = None,
        **kwargs,
    ) -> LatentSample:
        """
        Sample hierarchical infections for all subpopulations.

        Generates baseline Rt, subpopulation deviations with sum-to-zero
        constraint, initial infections, and runs n_subpops independent renewal processes.

        Parameters
        ----------
        n_days_post_init : int
            Number of days to simulate after initialization period
        subpop_fractions : ArrayLike
            Population fractions for all subpopulations. Shape: (n_subpops,).
            Must sum to 1.0.
        **kwargs
            Additional arguments (unused, for compatibility)

        Returns
        -------
        LatentSample
            Named tuple with fields:
            - aggregate: shape (n_total_days,)
            - all_subpops: shape (n_total_days, n_subpops)
        """
        # Parse and validate population structure
        pop = self._parse_and_validate_fractions(
            subpop_fractions=subpop_fractions,
        )

        n_total_days = self.n_initialization_points + n_days_post_init

        initial_log_rt = self.initial_log_rt_rv()

        log_rt_baseline = self.baseline_rt_process.sample(
            n_timepoints=n_total_days,
            initial_value=initial_log_rt,
            name_prefix="log_rt_baseline",
        )

        deviations_raw = self.subpop_rt_deviation_process.sample(
            n_timepoints=n_total_days,
            n_processes=pop.n_subpops,
            initial_value=jnp.zeros(pop.n_subpops),
            name_prefix="subpop_deviations",
        )

        # Sum-to-zero constraint ensures identifiability
        mean_deviation = jnp.mean(deviations_raw, axis=1, keepdims=True)
        deviations = deviations_raw - mean_deviation

        log_rt_subpop = log_rt_baseline + deviations
        rt_subpop = jnp.exp(log_rt_subpop)

        gen_int = self.gen_int_rv()

        I0 = jnp.asarray(self.I0_rv())
        self._validate_I0(I0)

        if I0.ndim == 0:
            I0_subpop = jnp.full(pop.n_subpops, I0)
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

        gen_int_reversed = jnp.flip(gen_int)
        recent_I0_all = I0_all[-gen_int.size :, :]

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
            infections_all * pop.fractions[jnp.newaxis, :], axis=1
        )

        self._validate_output_shapes(
            infections_aggregate,
            infections_all,
            n_total_days,
            pop,
        )

        # Record key quantities for diagnostics and posterior analysis
        with numpyro.handlers.scope(prefix=self.name):
            numpyro.deterministic("I0_init_all_subpops", I0_all)
            numpyro.deterministic("log_rt_baseline", log_rt_baseline)
            numpyro.deterministic("rt_baseline", jnp.exp(log_rt_baseline))
            numpyro.deterministic("rt_subpop", rt_subpop)
            numpyro.deterministic("subpop_deviations", deviations)
            numpyro.deterministic("infections_aggregate", infections_aggregate)

        return LatentSample(
            aggregate=infections_aggregate,
            all_subpops=infections_all,
        )
