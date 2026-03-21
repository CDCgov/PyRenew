"""
Shared latent infection process renewal model.
"""

from __future__ import annotations

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


class SharedInfections(BaseLatentInfectionProcess):
    """
    A single $\mathcal{R}(t)$ trajectory drives one renewal equation.

    Mathematical form:
    - $\log \mathcal{R}(t) \sim \text{TemporalProcess}$ (e.g., AR(1), RandomWalk)
    - $I(t) = \mathcal{R}(t) \sum_\tau I(t-\tau) \, g(\tau)$
    - Each observation: $\mu_k(t) = \alpha_k \sum_s I(t-s) \, \pi_k(s)$

    The constructor specifies model structure (priors, temporal processes).

    Parameters
    ----------
    gen_int_rv
        Generation interval PMF
    I0_rv
        Initial infection prevalence (proportion of population) at first
        observation time. Must return a scalar value in the interval (0, 1).
        Full I0 vector generated via exponential backprojection during sampling.
    shared_rt_process
        Temporal process for shared Rt dynamics
    initial_log_rt_rv
        Initial value for log(Rt) at time 0.  Can be estimated from data
        or given a prior distribution.
    n_initialization_points
        Number of initialization days before day 0. Must be at least
        ``len(gen_int_rv())`` to provide enough history for the renewal
        equation convolution. When using PyrenewBuilder, this is computed
        automatically from all observation processes.

    Notes
    -----
    When using PyrenewBuilder (recommended), n_initialization_points is computed
    automatically from all observation processes.
    """

    def __init__(
        self,
        *,
        gen_int_rv: RandomVariable,
        I0_rv: RandomVariable,
        shared_rt_process: TemporalProcess,
        initial_log_rt_rv: RandomVariable,
        n_initialization_points: int,
        name: str = "latent_infections",
    ) -> None:
        """
        Initialize shared infections process.

        Parameters
        ----------
        gen_int_rv
            Generation interval PMF
        I0_rv
            Initial infection prevalence (proportion of population)
        shared_rt_process
            Temporal process for shared Rt dynamics
        initial_log_rt_rv
            Initial value for log(Rt) at time 0.
        n_initialization_points
            Number of initialization days before day 0.
        name
            Name prefix for numpyro sample sites. All deterministic
            quantities are recorded under this scope (e.g.,
            ``"{name}::rt_shared"``). Default: ``"latent_infections"``.

        Raises
        ------
        ValueError
            If required parameters are missing or invalid
        """
        super().__init__(
            name=name,
            gen_int_rv=gen_int_rv,
            n_initialization_points=n_initialization_points,
        )

        if I0_rv is None:
            raise ValueError("I0_rv is required")
        self.I0_rv = I0_rv

        if isinstance(I0_rv, DeterministicVariable):
            self._validate_I0(I0_rv.value)

        if initial_log_rt_rv is None:
            raise ValueError("initial_log_rt_rv is required")
        self.initial_log_rt_rv = initial_log_rt_rv

        if shared_rt_process is None:
            raise ValueError("shared_rt_process is required")
        self.shared_rt_process = shared_rt_process

    def default_subpop_fractions(self) -> ArrayLike:
        """
        Return default population fractions for a single-population model.

        Returns
        -------
        ArrayLike
            ``jnp.array([1.0])``
        """
        return jnp.array([1.0])

    def validate(self) -> None:
        """
        Validate shared infections parameters.

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
        subpop_fractions: ArrayLike | None = None,
        **kwargs: object,
    ) -> LatentSample:
        """
        Sample shared infections using a single renewal process.

        Generates a shared Rt trajectory, computes initial infections via
        exponential backprojection, and runs one renewal equation.

        Parameters
        ----------
        n_days_post_init
            Number of days to simulate after initialization period
        subpop_fractions
            Population fractions. Defaults to ``[1.0]`` (single population).
            Must be ``[1.0]`` if provided.
        **kwargs
            Additional arguments (unused, for compatibility)

        Returns
        -------
        LatentSample
            Named tuple with fields:
            - aggregate: shape (n_total_days,)
            - all_subpops: shape (n_total_days, 1)
        """
        pop = self._parse_and_validate_fractions(
            subpop_fractions=subpop_fractions,
        )

        n_total_days = self.n_initialization_points + n_days_post_init

        initial_log_rt = self.initial_log_rt_rv()

        log_rt_shared = self.shared_rt_process.sample(
            n_timepoints=n_total_days,
            initial_value=initial_log_rt,
            name_prefix="log_rt_shared",
        )

        rt_shared = jnp.exp(log_rt_shared)

        gen_int = self.gen_int_rv()

        I0 = jnp.asarray(self.I0_rv())
        self._validate_I0(I0)

        initial_r = r_approx_from_R(
            R=rt_shared[0, 0],
            g=gen_int,
            n_newton_steps=4,
        )

        time_indices = jnp.arange(self.n_initialization_points)
        I0_init = I0 * jnp.exp(initial_r * time_indices)

        gen_int_reversed = jnp.flip(gen_int)
        recent_I0 = I0_init[-gen_int.size :]

        post_init_infections = compute_infections_from_rt(
            I0=recent_I0,
            Rt=rt_shared[self.n_initialization_points :, 0],
            reversed_generation_interval_pmf=gen_int_reversed,
        )

        infections_1d = jnp.concatenate([I0_init, post_init_infections])
        infections_all = infections_1d[:, jnp.newaxis]
        infections_aggregate = infections_1d

        self._validate_output_shapes(
            infections_aggregate,
            infections_all,
            n_total_days,
            pop,
        )

        with numpyro.handlers.scope(prefix=self.name, divider="::"):
            numpyro.deterministic("I0_init", I0_init)
            numpyro.deterministic("log_rt_shared", log_rt_shared)
            numpyro.deterministic("rt_shared", rt_shared)
            numpyro.deterministic("infections_aggregate", infections_aggregate)

        return LatentSample(
            aggregate=infections_aggregate,
            all_subpops=infections_all,
        )
