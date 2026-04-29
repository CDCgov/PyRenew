"""
Populaton-level single-trajectory latent infection process renewal model.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from numpyro.util import not_jax_tracer

from pyrenew.arrayutils import require_shape
from pyrenew.deterministic import DeterministicVariable
from pyrenew.distutil import validate_discrete_dist_vector
from pyrenew.latent.base import (
    BaseLatentInfectionProcess,
    LatentSample,
    PopulationStructure,
)
from pyrenew.latent.infection_functions import compute_infections_from_rt
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.math import r_approx_from_R
from pyrenew.metaclass import RandomVariable


class PopulationInfections(BaseLatentInfectionProcess):
    """
    A single $\\mathcal{R}(t)$ trajectory drives one renewal equation.

    The constructor specifies model structure (priors, temporal processes).
    """

    def __init__(
        self,
        *,
        name: str,
        gen_int_rv: RandomVariable,
        n_initialization_points: int,
        I0_rv: RandomVariable,
        single_rt_process: TemporalProcess,
        log_rt_time_0_rv: RandomVariable,
    ) -> None:
        """
        Initialize population-level infections process.

        Parameters
        ----------
        name
            Name prefix for numpyro sample sites. All deterministic
            quantities are recorded under this scope (e.g.,
            ``"{name}::rt_single"``).
        gen_int_rv
            Generation interval PMF
        n_initialization_points
            Number of initialization days before day 0.
        I0_rv
            Initial infection prevalence (proportion of population)
        single_rt_process
            Temporal process for single $\\mathcal{R}(t)$ dynamics
        log_rt_time_0_rv
            Initial value for log($\\mathcal{R}(t)$) at time 0.

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

        if log_rt_time_0_rv is None:
            raise ValueError("log_rt_time_0_rv is required")
        self.log_rt_time_0_rv = log_rt_time_0_rv

        if single_rt_process is None:
            raise ValueError("single_rt_process is required")
        self.single_rt_process = single_rt_process

    def default_subpop_fractions(self) -> ArrayLike:
        """
        Return default population fractions for a single-population model.

        Returns
        -------
        ArrayLike
            ``jnp.array([1.0])``
        """
        return jnp.array([1.0])

    def _validate_and_prepare_I0(
        self,
        I0: ArrayLike,
        pop: PopulationStructure,
    ) -> ArrayLike:
        """
        Validate that I0 is a scalar prevalence value.

        PopulationInfections operates on a single population, so I0 must be
        a scalar (0-dimensional array).

        Parameters
        ----------
        I0
            Initial infection prevalence from I0_rv, as a JAX array.
        pop
            Parsed population structure (unused, required by interface).

        Returns
        -------
        ArrayLike
            Validated scalar I0.

        Raises
        ------
        ValueError
            If I0 is not scalar or not in the interval (0, 1].
        """
        if I0.ndim != 0:
            raise ValueError(
                "PopulationInfections requires I0_rv to return a scalar prevalence"
            )
        return super()._validate_and_prepare_I0(I0, pop)

    def validate(self) -> None:
        """
        Validate population infections parameters.

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
        subpop_fractions: ArrayLike | None = None,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> LatentSample:
        """
        Sample population infections using a single renewal process.

        Generates a single daily $\\mathcal{R}(t)$ trajectory, computes initial
        infections via exponential backprojection, and runs one deterministic
        daily renewal equation. The temporal process may sample parameters at
        any supported cadence (for example daily or weekly/stepwise), but it
        must return a daily-length trajectory before the renewal equation is
        evaluated.

        Parameters
        ----------
        n_days_post_init
            Number of days to simulate after initialization period
        subpop_fractions
            Population fractions. Defaults to ``[1.0]`` (single population).
            Must be ``[1.0]`` if provided.
        first_day_dow
            Day of week for element 0 of the full latent infection time axis,
            including initialization days. When this latent process is used
            inside MultiSignalModel, the caller passes obs_start_date to the
            model and it converts the first observation day to this axis-origin
            day of week by subtracting n_initialization_points. Forwarded to
            ``single_rt_process``. See [pyrenew.latent.TemporalProcess][].
        **kwargs
            Additional arguments (unused, for compatibility)

        Returns
        -------
        LatentSample
            Named tuple with fields:
            - aggregate: shape (n_total_days,)
            - all_subpops: shape (n_total_days, 1)

        Raises
        ------
        ValueError
            If ``subpop_fractions`` does not represent a single population
            with fraction ``[1.0]`` or if ``I0_rv`` does not return a scalar.
        """
        pop = self._parse_and_validate_fractions(
            subpop_fractions=subpop_fractions,
        )
        frac_check = jnp.isclose(pop.fractions[0], 1.0, atol=1e-6)
        if pop.n_subpops != 1 or (not_jax_tracer(frac_check) and not frac_check):
            raise ValueError(
                "PopulationInfections requires exactly one subpopulation "
                "with fraction [1.0]"
            )

        n_total_days = self.n_initialization_points + n_days_post_init

        initial_log_rt = self.log_rt_time_0_rv()

        log_rt_single = self.single_rt_process.sample(
            n_timepoints=n_total_days,
            initial_value=initial_log_rt,
            name_prefix="log_rt_single",
            first_day_dow=first_day_dow,
        )
        require_shape(log_rt_single, (n_total_days, 1), "single_rt_process")

        rt_single = jnp.exp(log_rt_single)

        gen_int = self.gen_int_rv()

        I0 = self._validate_and_prepare_I0(jnp.asarray(self.I0_rv()), pop)

        initial_r = r_approx_from_R(
            R=rt_single[0, 0],
            g=gen_int,
            n_newton_steps=4,
        )

        time_indices = jnp.arange(self.n_initialization_points)
        I0_init = I0 * jnp.exp(initial_r * time_indices)

        gen_int_reversed = jnp.flip(gen_int)
        recent_I0 = I0_init[-gen_int.size :]

        post_init_infections = compute_infections_from_rt(
            I0=recent_I0,
            Rt=rt_single[self.n_initialization_points :, 0],
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
            numpyro.deterministic("log_rt_single", log_rt_single)
            numpyro.deterministic("rt_single", rt_single)
            numpyro.deterministic("infections_aggregate", infections_aggregate)

        return LatentSample(
            aggregate=infections_aggregate,
            all_subpops=infections_all,
        )
