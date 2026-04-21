"""
Model builder for multi-signal renewal models.

Automatically computes n_initialization_points from observation processes
and constructs properly configured models.
"""

from __future__ import annotations

from typing import Any

from pyrenew.latent.base import BaseLatentInfectionProcess
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.model.multisignal_model import MultiSignalModel
from pyrenew.observation.base import BaseObservationProcess

# Parameters that should be passed at sample time, not configure time
_SAMPLE_TIME_PARAMS = {
    "subpop_fractions",
}


class PyrenewBuilder:
    """
    Builder for multi-signal renewal models.

    Automatically computes n_initialization_points from observation
    processes and constructs a properly configured model.

    The builder pattern ensures that:
    1. n_initialization_points is computed correctly from all components
    2. Validation happens at build time (fail-fast)
    3. Latent infections are routed to the correct observations
    4. The API is clean and easy to use

    Population structure (subpop_fractions) is passed at sample/fit time, not at
    configure time. This allows a single model to be fit to multiple jurisdictions
    with different population structures.
    """

    def __init__(self) -> None:
        """
        Initialize a new model builder.

        The builder starts empty and is configured by calling:
        1. configure_latent() - set up the latent infection process
        2. add_observation() - add one or more observation processes
        3. build() - construct the final model
        """
        self.latent_class: type[BaseLatentInfectionProcess] | None = None
        self.latent_params: dict[str, Any] = {}
        self.observations: dict[str, BaseObservationProcess] = {}

    def configure_latent(
        self,
        latent_class: type[BaseLatentInfectionProcess],
        **params: dict[str, object],
    ) -> PyrenewBuilder:
        """
        Configure the latent infection process.

        Parameters
        ----------
        latent_class
            Class to use for latent infections (e.g., PopulationInfections,
            SubpopulationInfections, or a custom implementation)
        **params
            Parameters for latent class constructor (model structure).
            DO NOT include n_initialization_points - it will be computed
            automatically from observation processes.
            DO NOT include population structure params (subpop_fractions) -
            these are passed at sample/fit time to allow fitting to multiple
            jurisdictions.

        Returns
        -------
        PyrenewBuilder
            Self, for method chaining

        Raises
        ------
        ValueError
            If n_initialization_points or population structure params are included
        RuntimeError
            If latent has already been configured
        """
        if self.latent_class is not None:
            raise RuntimeError(
                "Latent process already configured. Create a new builder for "
                "a different configuration."
            )

        if "n_initialization_points" in params:
            raise ValueError(
                "Do not specify n_initialization_points - it will be computed "
                "automatically from observation processes and generation interval. "
                "Use PyrenewBuilder.build() to create the model with the correct value."
            )

        # Check for population structure params that should be at sample time
        sample_time_found = _SAMPLE_TIME_PARAMS.intersection(params.keys())
        if sample_time_found:
            raise ValueError(
                f"Do not specify {sample_time_found} at configure time. "
                f"Population structure is passed at sample/fit time to allow "
                f"fitting the same model to multiple jurisdictions."
            )

        self.latent_class = latent_class
        self.latent_params = params
        return self

    def add_observation(
        self,
        obs_process: BaseObservationProcess,
    ) -> PyrenewBuilder:
        """
        Add an observation process to the model.

        The observation process's ``name`` attribute is used as the unique
        identifier for this observation. This name is used when passing
        observation data to ``model.sample()`` and ``model.fit()``, and also
        prefixes all numpyro sample sites created by the process.

        Parameters
        ----------
        obs_process
            Configured observation process instance (e.g., PopulationCounts,
            Wastewater, SubpopulationCounts). Must have a ``name`` attribute.

        Returns
        -------
        PyrenewBuilder
            Self, for method chaining

        Raises
        ------
        ValueError
            If an observation with this name already exists
        """
        name = obs_process.name
        if name in self.observations:
            raise ValueError(
                f"Observation '{name}' already added. "
                f"Each observation must have a unique name."
            )

        self.observations[name] = obs_process
        return self

    def compute_n_initialization_points(self) -> int:
        """
        Compute required n_initialization_points from all components.

        Formula: n_initialization_points = max(all lookbacks)

        Where lookbacks include:
        - Generation interval length (from latent process)
        - All observation delay/shedding PMF lengths

        Useful for inspection before building the model.

        Returns
        -------
        int
            Minimum n_initialization_points needed to satisfy all components

        Raises
        ------
        ValueError
            If latent process not configured or gen_int_rv missing
            If any observation process doesn't implement get_required_lookback()
        """
        if self.latent_class is None:
            raise ValueError(
                "Must call configure_latent() before computing n_initialization_points"
            )

        # Get generation interval length from latent params
        gen_int_rv = self.latent_params.get("gen_int_rv")
        if gen_int_rv is None:
            raise ValueError("gen_int_rv is required in latent process parameters")

        # Start with generation interval lookback
        lookbacks = [len(gen_int_rv())]

        # Add lookback from each observation process
        for name, obs_process in self.observations.items():
            lookbacks.append(obs_process.lookback_days())

        # Formula: max(all lookbacks)
        # For generation interval (1-indexed): L-element PMF has max lag L days → need L init points
        # For delay distributions (0-indexed): L-element PMF has max delay L-1 days
        # We need at least max(all lookbacks) to satisfy the renewal equation extraction
        n_init = max(lookbacks)

        return n_init

    def _validate_coherence(self) -> None:
        """
        Enforce end-to-end coherence between R(t) cadence and observation cadences.

        Called at the start of ``build()`` before any model components are
        constructed. Inspects ``self.observations`` for ``aggregation_period``
        and ``period_end_dow`` attributes, and walks ``self.latent_params``
        values for any ``TemporalProcess`` instances to read their
        ``step_size`` attribute.

        Checks:

        - All observations sharing the same ``aggregation_period > 1`` must
          agree on ``period_end_dow``.
        - Every temporal-process ``step_size`` must be ``<=`` the finest
          observation ``aggregation_period``.
        - If a temporal process has ``step_size > 1``, that ``step_size``
          must equal the ``aggregation_period`` of every observation whose
          ``aggregation_period > 1``.

        Raises
        ------
        ValueError
            If any of the three rules is violated.
        """
        agg_by_dow: dict[int, set[int]] = {}
        agg_periods: list[int] = []
        for name, obs in self.observations.items():
            P = getattr(obs, "aggregation_period", 1)
            agg_periods.append(P)
            if P > 1:
                agg_by_dow.setdefault(P, set()).add(getattr(obs, "period_end_dow"))

        for P, dows in agg_by_dow.items():
            if len(dows) > 1:
                raise ValueError(
                    f"Observations with aggregation_period={P} must agree on "
                    f"period_end_dow; got values {sorted(dows)}"
                )

        temporal_processes = {
            name: value
            for name, value in self.latent_params.items()
            if isinstance(value, TemporalProcess)
        }

        finest = min(agg_periods) if agg_periods else 1
        coarse_periods = {P for P in agg_periods if P > 1}

        for param_name, process in temporal_processes.items():
            step_size = getattr(process, "step_size", 1)
            if step_size > finest:
                raise ValueError(
                    f"Temporal process '{param_name}' has step_size={step_size} "
                    f"exceeding the finest observation aggregation_period "
                    f"({finest}). Parameterize R(t) at least as finely as the "
                    f"finest observed signal."
                )
            if step_size > 1:
                mismatched = {P for P in coarse_periods if P != step_size}
                if mismatched:
                    raise ValueError(
                        f"Temporal process '{param_name}' has step_size="
                        f"{step_size} but observations include "
                        f"aggregation_period(s) {sorted(mismatched)} that do "
                        f"not match. All coarse cadences must agree."
                    )

    def build(self) -> MultiSignalModel:
        """
        Build the multi-signal model with computed n_initialization_points.

        This method:
        1. Enforces coherence between R(t) cadence and observation cadences
        2. Computes n_initialization_points from all components
        3. Constructs the latent process with the computed value
        4. Creates a MultiSignalModel with automatic infection routing
        5. Validates that observation/latent types are compatible

        Can be called multiple times to create multiple model instances.

        Returns
        -------
        MultiSignalModel
            Configured model ready for sampling

        Raises
        ------
        ValueError
            If latent process not configured, or if R(t) and observation
            cadences are incoherent.
        """
        if self.latent_class is None:
            raise ValueError("Must call configure_latent() before build()")

        self._validate_coherence()

        # Compute n_initialization_points
        n_init = self.compute_n_initialization_points()

        # Construct latent process with computed n_initialization_points
        latent_params = {**self.latent_params, "n_initialization_points": n_init}
        if "name" not in latent_params:
            latent_params["name"] = self.latent_class.__name__

        latent_process = self.latent_class(**latent_params)

        # Build model
        model = MultiSignalModel(
            latent_process=latent_process,
            observations=self.observations,
        )

        return model
