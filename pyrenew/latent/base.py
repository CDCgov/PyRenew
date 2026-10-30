"""
Base class for latent infection processes with subpopulation structure.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.util import not_jax_tracer

from pyrenew.latent.infection_functions import (
    compute_infections_from_rt,
    compute_infections_from_rt_with_feedback,
)
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.metaclass import RandomVariable


class LatentSample(NamedTuple):
    """
    Output from latent infection process sampling.

    Attributes
    ----------
    aggregate
        Total infections aggregated across all subpopulations.
        Shape: (n_total_days,)
    all_subpops
        Infections for all subpopulations.
        Shape: (n_total_days, n_subpops)
    """

    aggregate: ArrayLike
    all_subpops: ArrayLike


@dataclass
class PopulationStructure:
    """
    Parsed and validated population structure for a jurisdiction.

    Attributes
    ----------
    fractions
        Population fractions for all subpopulations.
        Shape: (n_subpops,)
    """

    fractions: ArrayLike

    @property
    def n_subpops(self) -> int:
        """
        Total number of subpopulations.

        Returns
        -------
        int
            The number of subpopulations.
        """
        return len(self.fractions)


def _validate_feedback_options(
    infection_feedback_strength_rv: RandomVariable | None,
    infection_feedback_pmf_rv: RandomVariable | None,
) -> None:
    """
    Check that the infection feedback options are given together.

    Parameters
    ----------
    infection_feedback_strength_rv
        Infection feedback strength, or None.
    infection_feedback_pmf_rv
        Infection feedback PMF, or None.

    Raises
    ------
    ValueError
        If exactly one of the two is None.
    """
    if (infection_feedback_strength_rv is None) != (infection_feedback_pmf_rv is None):
        given = (
            "infection_feedback_strength_rv"
            if infection_feedback_pmf_rv is None
            else "infection_feedback_pmf_rv"
        )
        raise ValueError(
            "infection_feedback_strength_rv and infection_feedback_pmf_rv must "
            f"be given together or both left out; got only {given}."
        )


def _compute_renewal(
    I0: ArrayLike,
    rt_raw: ArrayLike,
    gen_int: ArrayLike,
    infection_feedback_strength_rv: RandomVariable | None,
    infection_feedback_pmf_rv: RandomVariable | None,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute post-initialization infections and the adjusted Rt used to generate them.

    Parameters
    ----------
    I0
        Infections during the initialization period, shape ``(n_init,)``
        or ``(n_init, n_subpops)``. Only the last ``gen_int.size`` days
        are used.
    rt_raw
        Raw reproduction number after the initialization period, from the
        temporal processes. Shape ``(n_days,)`` or ``(n_days, n_subpops)``.
    gen_int
        Generation interval PMF.
    infection_feedback_strength_rv
        Infection feedback strength, or None for plain renewal.
    infection_feedback_pmf_rv
        Infection feedback PMF, or None for plain renewal.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        Post-initialization infections and the adjusted Rt used to
        generate them, both with the shape of ``rt_raw``. Without
        feedback, the adjusted Rt is ``rt_raw``.

    Raises
    ------
    ValueError
        If the feedback strength does not broadcast to the shape of
        ``rt_raw``, or the feedback PMF length differs from the generation
        interval length.
    """
    recent_I0 = I0[-gen_int.size :]
    reversed_gen_int = jnp.flip(gen_int)
    if infection_feedback_strength_rv is None or infection_feedback_pmf_rv is None:
        infections = compute_infections_from_rt(
            I0=recent_I0,
            Rt=rt_raw,
            reversed_generation_interval_pmf=reversed_gen_int,
        )
        return infections, rt_raw

    strength = jnp.atleast_1d(infection_feedback_strength_rv())
    try:
        strength = jnp.broadcast_to(strength, rt_raw.shape)
    except ValueError as e:
        raise ValueError(
            f"infection_feedback_strength_rv returned shape {strength.shape}, "
            f"which does not broadcast to the shape of Rt, {rt_raw.shape}."
        ) from e

    feedback_pmf = jnp.asarray(infection_feedback_pmf_rv())
    if feedback_pmf.shape != gen_int.shape:
        raise ValueError(
            "infection_feedback_pmf_rv must return an array with the same "
            f"length as the generation interval, {gen_int.size}; got shape "
            f"{feedback_pmf.shape}."
        )

    return compute_infections_from_rt_with_feedback(
        I0=recent_I0,
        Rt_raw=rt_raw,
        infection_feedback_strength=strength,
        reversed_generation_interval_pmf=reversed_gen_int,
        reversed_infection_feedback_pmf=jnp.flip(feedback_pmf),
    )


class BaseLatentInfectionProcess(RandomVariable):
    """
    Base class for latent infection processes with subpopulation structure.

    Provides common functionality for hierarchical and partitioned infection models:
    - Population fraction validation and parsing (at sample time)
    - Standard output structure via LatentSample

    All subclasses return infections as a ``LatentSample`` named tuple with fields:
    (aggregate, all_subpops). Observation processes are responsible for selecting
    which subpopulations they observe via indexing.

    The constructor specifies model structure (generation interval, priors,
    temporal processes). Population structure (subpop_fractions) is provided at
    sample time, allowing a single model to be fit to multiple jurisdictions.

    Parameters
    ----------
    gen_int_rv
        Generation interval PMF
    n_initialization_points
        Number of initialization days before the first observation day.
        Latent and observation arrays use a shared padded time axis with
        element 0 at the start of this initialization period. In observation
        natural coordinates, day 0 is the first observed data day; on the
        shared padded axis, that same day is index ``n_initialization_points``.
        Must be at least ``len(gen_int_rv())`` to provide enough history for
        the renewal equation convolution.

    Notes
    -----
    Population structure (subpop_fractions) is passed to the sample() method, not
    the constructor. This allows a single model instance to be fit to multiple
    datasets with different jurisdiction structures.

    When using PyrenewBuilder (recommended), n_initialization_points is computed
    automatically from all observation processes. When constructing latent processes
    directly, you must specify n_initialization_points explicitly.
    """

    def __init__(
        self,
        *,
        name: str,
        gen_int_rv: RandomVariable,
        n_initialization_points: int,
    ) -> None:
        """
        Initialize base latent infection process.

        Parameters
        ----------
        name
            A name for this random variable.
        gen_int_rv
            Generation interval PMF
        n_initialization_points
            Number of initialization days; see the class-level parameter
            documentation for the shared padded-axis convention.

        Raises
        ------
        ValueError
            If gen_int_rv is None or n_initialization_points is insufficient.
        """
        super().__init__(name=name)
        if gen_int_rv is None:
            raise ValueError("gen_int_rv is required")
        self.gen_int_rv = gen_int_rv

        gen_int_length = len(self.gen_int_rv())
        if n_initialization_points < gen_int_length:
            raise ValueError(
                f"n_initialization_points must be at least the generation "
                f"interval length ({gen_int_length}), got "
                f"{n_initialization_points}"
            )
        self.n_initialization_points = n_initialization_points

    def default_subpop_fractions(self) -> ArrayLike | None:
        """
        Return default population fractions, or None if caller must provide them.

        Override this function in order to omit specification of
        subpop_fractions at sample time, in which case it will be called
        by _parse_and_validate_fractions.
        Must return a valid array of positive elements that sums to 1.

        Returns
        -------
        ArrayLike or None
            Default fractions array, or None if no default exists.
        """
        return None

    def _parse_and_validate_fractions(
        self,
        subpop_fractions: ArrayLike | None = None,
    ) -> PopulationStructure:
        """
        Parse and validate population fraction parameters.

        Parameters
        ----------
        subpop_fractions
            Population fractions for all subpopulations. Must be a 1D array
            with at least one element. Values must be non-negative and sum to 1.

        Returns
        -------
        PopulationStructure
            Parsed and validated population structure

        Raises
        ------
        ValueError
            If fractions are invalid, don't sum to 1.0, or are None with
            no subclass default.
        """
        if subpop_fractions is None:
            subpop_fractions = self.default_subpop_fractions()
            if subpop_fractions is None:
                raise ValueError("subpop_fractions must be provided")

        fractions = jnp.asarray(subpop_fractions)

        if fractions.ndim != 1:
            raise ValueError("subpop_fractions must be a 1D array")

        if len(fractions) == 0:
            raise ValueError("Must have at least one subpopulation")

        # Only validate when results are concrete (not during JAX tracing)
        # Check the result of the comparison, not just the input, because
        # JIT compilation can trace operations on concrete arrays
        neg_check = jnp.any(fractions < 0)
        if not_jax_tracer(neg_check) and neg_check:
            raise ValueError("All population fractions must be non-negative")

        total = jnp.sum(fractions)
        sum_check = jnp.isclose(total, 1.0, atol=1e-6)
        if not_jax_tracer(sum_check) and not sum_check:
            raise ValueError(
                f"Population fractions must sum to 1.0, got {float(total):.6f}"
            )

        return PopulationStructure(
            fractions=fractions,
        )

    @staticmethod
    def _validate_output_shapes(
        infections_aggregate: ArrayLike,
        infections_all: ArrayLike,
        n_total_days: int,
        pop: PopulationStructure,
    ) -> None:
        """
        Validate that output shapes match expected dimensions.

        Parameters
        ----------
        infections_aggregate
            Aggregate infections (sum across subpopulations)
        infections_all
            All subpopulation infections
        n_total_days
            Expected number of days (n_initialization_points + n_days_post_init)
        pop
            Population structure

        Raises
        ------
        ValueError
            If any output shape is incorrect
        """
        expected = {
            "infections_aggregate": (n_total_days,),
            "infections_all": (n_total_days, pop.n_subpops),
        }

        actual = {
            "infections_aggregate": infections_aggregate.shape,
            "infections_all": infections_all.shape,
        }

        for name, expected_shape in expected.items():
            if actual[name] != expected_shape:
                raise ValueError(
                    f"{name} has incorrect shape. "
                    f"Expected {expected_shape}, got {actual[name]}"
                )

    @staticmethod
    def _validate_I0(I0: ArrayLike) -> None:
        """
        Validate initial infections per resident.

        I0 is new infections per resident on the first day of the
        initialization period. Values must be in the interval (0, 1].

        Parameters
        ----------
        I0
            New infections per resident on the first day of the
            initialization period (scalar or array).

        Raises
        ------
        ValueError
            If any I0 value is not in the interval (0, 1]

        Notes
        -----
        Validation is skipped during JAX tracing (e.g., when using Predictive)
        since traced values cannot be used in Python boolean operations.
        """
        I0 = jnp.asarray(I0)

        # Only validate when results are concrete (not during JAX tracing)
        # Check the result of the comparison, not just the input, because
        # JIT compilation can trace operations on concrete arrays
        pos_check = jnp.any(I0 <= 0)
        if not_jax_tracer(pos_check) and pos_check:
            raise ValueError(
                f"I0 must be positive (got min={float(jnp.min(I0)):.6f}). "
                "I0 is new infections per resident on the first day of "
                "the initialization period."
            )
        max_check = jnp.any(I0 > 1)
        if not_jax_tracer(max_check) and max_check:
            raise ValueError(
                f"I0 must be <= 1 (got max={float(jnp.max(I0)):.6f}). "
                "I0 is new infections per resident on the first day of "
                "the initialization period."
            )

    def _validate_and_prepare_I0(
        self,
        I0: ArrayLike,
        pop: PopulationStructure,
    ) -> ArrayLike:
        """
        Validate and prepare I0 for use in the renewal equation.

        Subclasses override this to enforce shape constraints (e.g., scalar
        for PopulationInfections) or broadcast I0 to match the population
        structure (e.g., scalar to per-subpop array for
        SubpopulationInfections).

        Parameters
        ----------
        I0
            New infections per resident on the first day of the
            initialization period, from I0_rv, as a JAX array.
        pop
            Parsed population structure.

        Returns
        -------
        ArrayLike
            Validated (and possibly reshaped) I0.

        Raises
        ------
        ValueError
            If I0 values are not in the interval (0, 1].
        """
        self._validate_I0(I0)
        return I0

    def get_required_lookback(self) -> int:
        """
        Return the generation interval length for builder pattern support.

        This method is used by PyrenewBuilder to compute n_initialization_points
        from all model components. Returns the generation interval PMF length.

        Returns
        -------
        int
            Length of generation interval PMF
        """
        return len(self.gen_int_rv())

    def requires_calendar_anchor(self) -> bool:
        """
        Report whether this latent process needs a calendar anchor at sample time.

        The default implementation inspects this instance's attributes
        for :class:`pyrenew.latent.temporal_processes.TemporalProcess`
        instances and returns ``True`` if any of them has
        ``requires_calendar_anchor=True``. Subclasses may override to add
        additional calendar-aligned components.

        Returns
        -------
        bool
            ``True`` if the caller of :meth:`sample` must supply a
            ``first_day_dow`` (derived from ``obs_start_date`` at the
            model entry point); ``False`` otherwise.
        """
        return any(
            isinstance(value, TemporalProcess)
            and getattr(value, "requires_calendar_anchor", False)
            for value in vars(self).values()
        )

    @abstractmethod
    def sample(
        self,
        n_days_post_init: int,
        subpop_fractions: ArrayLike | None = None,
        **kwargs: object,
    ) -> LatentSample:
        """
        Sample latent infections for all subpopulations.

        Parameters
        ----------
        n_days_post_init
            Number of days to simulate after initialization period
        subpop_fractions
            Population fractions for all subpopulations.
            Shape: (n_subpops,). Must sum to 1.0.
            If None, falls back to ``self.default_subpop_fractions()``.
        **kwargs
            Additional parameters required by specific implementations

        Returns
        -------
        LatentSample
            Named tuple with fields:
            - aggregate: shape (n_total_days,)
            - all_subpops: shape (n_total_days, n_subpops)

            where n_total_days = n_initialization_points + n_days_post_init
        """
        pass  # pragma: no cover
