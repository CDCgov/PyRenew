# numpydoc ignore=GL08
"""
Observation processes for connecting infections to observed data.

``BaseObservationProcess`` is the abstract base. Concrete subclasses:

- ``PopulationCounts``: Aggregate counts (admissions, deaths)
- ``SubpopulationCounts``: Subpopulation-level counts
- ``MeasurementObservation``: Continuous subpopulation-level signals (e.g., wastewater)

All observation processes implement:

- ``sample()``: Sample observations given infections
- ``infection_resolution()``: returns ``"aggregate"`` or ``"subpop"``
- ``lookback_days()``: returns required infection history length

Noise models (``CountNoise``, ``MeasurementNoise``) are composable—pass them
to observation constructors to control the output distribution.
"""

from pyrenew.observation.base import BaseObservationProcess
from pyrenew.observation.count_observations import (
    CountObservation,
    PopulationCounts,
    SubpopulationCounts,
)
from pyrenew.observation.measurement_observations import MeasurementObservation
from pyrenew.observation.negativebinomial import NegativeBinomialObservation
from pyrenew.observation.noise import (
    CountNoise,
    HierarchicalNormalNoise,
    MeasurementNoise,
    NegativeBinomialNoise,
    PoissonNoise,
)
from pyrenew.observation.types import ObservationSample

__all__ = [
    # Existing (kept for backward compatibility)
    "NegativeBinomialObservation",
    # Base classes and types
    "BaseObservationProcess",
    "ObservationSample",
    # Noise models
    "CountNoise",
    "PoissonNoise",
    "NegativeBinomialNoise",
    "MeasurementNoise",
    "HierarchicalNormalNoise",
    # Observation processes
    "BaseObservationProcess",
    "CountObservation",
    "MeasurementObservation",
    "PopulationCounts",
    "SubpopulationCounts",
]
