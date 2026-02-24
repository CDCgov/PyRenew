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
        Number of initialization days before day 0. Must be at least
        ``len(gen_int_rv())`` to provide enough history for the renewal
        equation convolution.

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
            Number of initialization days before day 0. Must be at least
            ``len(gen_int_rv())`` to provide enough history for the renewal
            equation convolution.

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

    @staticmethod
    def _parse_and_validate_fractions(
        subpop_fractions: ArrayLike = None,
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
            If fractions are invalid or don't sum to 1.0
        """
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
        Validate that I0 values are valid infection prevalences.

        I0 represents initial infection prevalence as a proportion of the
        population. Values must be in the interval (0, 1].

        Parameters
        ----------
        I0
            Initial infection prevalence (scalar or array)

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
                "I0 represents infection prevalence as a proportion of the population."
            )
        max_check = jnp.any(I0 > 1)
        if not_jax_tracer(max_check) and max_check:
            raise ValueError(
                f"I0 must be <= 1 (got max={float(jnp.max(I0)):.6f}). "
                "I0 represents infection prevalence as a proportion of the population."
            )

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

    @abstractmethod
    def validate(self) -> None:
        """
        Validate latent process parameters.

        Subclasses must implement this method to validate all parameters specific
        to their implementation (e.g., temporal process parameters, I0 parameters).

        Common validation (n_initialization_points, gen_int_rv) is performed in
        __init__. Population structure validation is performed at sample time.

        Raises
        ------
        ValueError
            If any parameters fail validation
        """
        pass  # pragma: no cover

    @abstractmethod
    def sample(
        self,
        n_days_post_init: int,
        *,
        subpop_fractions: ArrayLike = None,
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
