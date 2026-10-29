"""
Subpopulation latent infections coupled by a fixed mobility matrix.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike

from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent.base import PopulationStructure
from pyrenew.latent.infection_functions import (
    compute_infections_from_rt_with_mobility,
)
from pyrenew.latent.subpopulation_infections import BaseSubpopulationInfections
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.metaclass import RandomVariable


class MobilitySubpopulationInfections(BaseSubpopulationInfections):
    r"""
    Subpopulation renewal model coupled by a fixed mobility matrix.

    Subpopulations are home regions whose residents spend part of each day
    at other locations. Transmission happens at each location, at that
    location's reproduction number, among the people present there, and
    each infection is assigned to the home region of the person infected.
    This implements Equation 2 of Roy, Clapham, and Mishra (2025),
    [Incorporating human mobility to enhance epidemic response and estimate
    real-time reproduction numbers](https://doi.org/10.1371/journal.pcbi.1013642).
    With depletion enabled, PyRenew additionally caps each day's new
    infections at the remaining susceptible population.

    Infections are per resident: ``infections[t, k]`` is new infections
    among residents of home region ``k`` on day ``t``, divided by the number
    of residents of ``k``. With ``M`` the mobility matrix and
    ``p = subpop_fractions``, one day with depletion enabled is:

    ```python
    present = M @ p
    share_present = M * p[None, :] / jnp.where(present > 0, present, 1.0)[:, None]
    infectiousness = reversed_gen_int @ history
    susceptible = 1.0 - cumulative
    new_infections_raw = M.T @ (rt * (share_present @ infectiousness))
    new_infections = jnp.minimum(susceptible * new_infections_raw, susceptible)
    ```

    ``share_present[j, k]`` is the share of the people present at location
    ``j`` who live in region ``k``, and ``cumulative`` counts every infection
    since the first day of the initialization period. Initial infections
    must be finite and non-negative. With depletion enabled, their sum
    over all initialization days must be at most 1 per resident in each
    region. Concrete invalid values raise ``ValueError``; during JAX
    tracing, callers must ensure sampled initialization meets these bounds.

    ``M[j, k]`` is the fraction of residents of home region ``k`` present at
    location ``j``: columns are home regions, rows are locations, and each
    column sums to 1. Regions must be in the same order in the matrix,
    ``subpop_fractions``, a vector-valued ``I0_rv``, a vector-valued
    ``log_rt_time_0_rv``, and every observation's ``subpop_indices``.

    The recorded values are those of
    [pyrenew.latent.SubpopulationInfections][], plus
    ``susceptible_fraction``. Under mobility:

    - ``rt_subpop[t, j]`` is the reproduction number at location ``j``,
      where transmission happens, when everyone is susceptible.
    - ``rt_subpop_adjusted`` is
      ``rt_subpop * (share_present @ susceptible_fraction)`` after the
      initialization period, with depletion enabled. This adjusts for
      susceptibility but does not account for the incidence ceiling. It
      equals new infections per unit of infectiousness present only when
      the ceiling does not bind. It is 0 at a location with no one present.
    - ``susceptible_fraction[t, i]`` is the share of residents of home
      region ``i`` not infected before day ``t``. It is recorded as 1 minus
      cumulative infections and can become negative with depletion off.

    Susceptible depletion is on by default, as in Eq 2. With it off,
    neither susceptibility scaling nor the incidence ceiling is applied,
    and inferred raw values of ``rt_subpop`` will need to directly absorb
    any reductions in transmission due to susceptible depletion.

    The matrix is fixed. Fitting and prediction use the matrix the model
    was built with; it is available as ``mobility_matrix_rv.value``.
    """

    def __init__(
        self,
        *,
        name: str,
        gen_int_rv: RandomVariable,
        n_initialization_points: int,
        I0_rv: RandomVariable,
        baseline_rt_process: TemporalProcess,
        subpop_rt_deviation_process: TemporalProcess,
        log_rt_time_0_rv: RandomVariable,
        mobility_matrix_rv: DeterministicVariable,
        susceptible_depletion: bool = True,
    ) -> None:
        """
        Initialize the mobility-coupled infections process.

        Parameters
        ----------
        name
            Name prefix for numpyro sample sites. All deterministic
            quantities are recorded under this scope (e.g.,
            ``"{name}::rt_subpop"``, ``"{name}::rt_subpop_adjusted"``, and
            ``"{name}::susceptible_fraction"``).
        gen_int_rv
            Generation interval PMF
        n_initialization_points
            Number of initialization days before day 0.
        I0_rv
            New infections per resident on the first day of the
            initialization period, the earliest day on the time axis. This
            is not prevalence. Returns either a scalar, one value for all
            regions, or an array of shape ``(n_subpops,)``, one value per
            region in the order of ``subpop_fractions``. Every
            initialization day counts toward susceptible depletion. With
            depletion enabled, the sum over all initialization days must
            be at most 1 per resident in each region.
        baseline_rt_process
            Temporal process for baseline $\\mathcal{R}(t)$ dynamics
        subpop_rt_deviation_process
            Temporal process for subpopulation deviations
        log_rt_time_0_rv
            Log reproduction number on the first post-initialization day.
            Returns either a scalar, one value for
            all regions, or an array of shape ``(n_subpops,)``, one value per
            region in the order of ``subpop_fractions``, which gives each
            region its own starting $\\mathcal{R}(t)$. Each region's starting
            Rt determines its exponential initialization growth. Temporal
            processes cover only post-initialization days; daily Rt records
            prepend copies of the starting values for initialization.
        mobility_matrix_rv
            The mobility matrix used on every day, as a
            ``DeterministicVariable`` whose value has shape
            ``(n_subpops, n_subpops)``. ``mobility_matrix[j, k]`` is the
            fraction of residents of home region ``k`` present at location
            ``j``; each column sums to 1. Rows and columns are in the order
            of ``subpop_fractions``. Convert a data frame with
            ``.to_numpy()``. Transpose a matrix whose rows sum to 1.
        susceptible_depletion
            If True (default), new infections among residents of each region
            are scaled by the fraction of those residents not yet infected,
            as in Roy et al. Eq 2, and capped at that fraction. If False,
            neither scaling nor the ceiling is applied, and ``rt_subpop``
            absorbs depletion.

        Raises
        ------
        TypeError
            If mobility_matrix_rv is not a DeterministicVariable.
        ValueError
            If required parameters are missing or invalid, or if the
            mobility matrix is not a valid column-stochastic matrix.
        """
        super().__init__(
            name=name,
            gen_int_rv=gen_int_rv,
            n_initialization_points=n_initialization_points,
            I0_rv=I0_rv,
            baseline_rt_process=baseline_rt_process,
            subpop_rt_deviation_process=subpop_rt_deviation_process,
            log_rt_time_0_rv=log_rt_time_0_rv,
        )
        if not isinstance(mobility_matrix_rv, DeterministicVariable):
            raise TypeError(
                "mobility_matrix_rv must be a DeterministicVariable; the "
                "mobility matrix is data and cannot be random. Got "
                f"{type(mobility_matrix_rv).__name__}."
            )
        self._validate_mobility_matrix(jnp.asarray(mobility_matrix_rv.value))
        self.mobility_matrix_rv = mobility_matrix_rv
        self.susceptible_depletion = susceptible_depletion

    @staticmethod
    def _validate_mobility_matrix(matrix: ArrayLike) -> None:
        """
        Check that the mobility matrix is a valid column-stochastic matrix.

        Parameters
        ----------
        matrix
            The mobility matrix, as a JAX array.

        Raises
        ------
        ValueError
            If the matrix is not square and two-dimensional, has non-finite
            or negative entries, or has a column that does not sum to 1.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "mobility_matrix_rv must be a single square (n, n) matrix; "
                f"got shape {matrix.shape}."
            )
        if not jnp.all(jnp.isfinite(matrix)):
            raise ValueError("mobility_matrix_rv must have only finite entries.")
        if jnp.any(matrix < 0):
            raise ValueError("mobility_matrix_rv must have only non-negative entries.")
        column_sums = jnp.sum(matrix, axis=0)
        if not jnp.all(jnp.isclose(column_sums, 1.0, atol=1e-6)):
            message = (
                "Each column of mobility_matrix_rv must sum to 1; got column "
                f"sums {column_sums.tolist()}."
            )
            if jnp.all(jnp.isclose(jnp.sum(matrix, axis=1), 1.0, atol=1e-6)):
                message += (
                    " The rows sum to 1; mobility_matrix[j, k] must be the "
                    "fraction of residents of k present at j. Did you mean to pass the transpose?"
                )
            raise ValueError(message)

    def _parse_and_validate_fractions(
        self,
        subpop_fractions: ArrayLike | None = None,
    ) -> PopulationStructure:
        """
        Parse population fractions and check them against the mobility matrix.

        Parameters
        ----------
        subpop_fractions
            Population fractions for all subpopulations.

        Returns
        -------
        PopulationStructure
            Parsed and validated population structure.

        Raises
        ------
        ValueError
            If the fractions are invalid, or if the matrix does not have one
            row and one column per subpopulation.
        """
        pop = super()._parse_and_validate_fractions(subpop_fractions)
        matrix = jnp.asarray(self.mobility_matrix_rv.value)
        if matrix.shape != (pop.n_subpops, pop.n_subpops):
            raise ValueError(
                f"mobility_matrix_rv has shape {matrix.shape}, but "
                f"subpop_fractions has {pop.n_subpops} entries; the matrix "
                f"must have shape ({pop.n_subpops}, {pop.n_subpops})."
            )
        return pop

    def _infections_from_rt(
        self,
        rt_raw: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        pop: PopulationStructure,
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Compute post-initialization infections and adjusted Rt with mobility coupling.

        Parameters
        ----------
        rt_raw
            Raw Rt at each location after the initialization period.
        I0
            Infections per resident on every initialization day.
        gen_int
            Generation interval PMF.
        pop
            Parsed population structure.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            Post-initialization infections per resident and the reproduction
            number at each location after susceptibility adjustment,
            before the incidence ceiling.
        """
        post_init_infections, rt_adjusted = compute_infections_from_rt_with_mobility(
            I0=I0,
            Rt_raw=rt_raw,
            reversed_generation_interval_pmf=jnp.flip(gen_int),
            mobility_matrix=jnp.asarray(self.mobility_matrix_rv()),
            subpop_fractions=pop.fractions,
            susceptible_depletion=self.susceptible_depletion,
        )
        infections_all = jnp.vstack([I0, post_init_infections])
        infected_before = jnp.vstack(
            [
                jnp.zeros((1, infections_all.shape[1])),
                jnp.cumsum(infections_all, axis=0)[:-1],
            ]
        )
        susceptible_fraction = 1.0 - infected_before
        with numpyro.handlers.scope(prefix=self.name, divider="::"):
            numpyro.deterministic("susceptible_fraction", susceptible_fraction)
        return post_init_infections, rt_adjusted
