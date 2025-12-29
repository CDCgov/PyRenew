# numpydoc ignore=GL08
"""
Observation processes for connecting infections to observed data.

Architecture
------------
``BaseObservationProcess`` is the abstract base. Concrete subclasses:

- ``Counts``: Aggregate counts (admissions, deaths)
- ``CountsBySubpop``: Subpopulation-level counts
- ``Measurements``: Continuous subpopulation-level signals (e.g., wastewater)

All observation processes implement:

- ``sample()``: Sample observations given infections
- ``infection_resolution()``: returns ``"aggregate"`` or ``"subpop"``
- ``lookback_days()``: returns required infection history length

Noise models (``CountNoise``, ``MeasurementNoise``) are composable—pass them
to observation constructors to control the output distribution.
"""

from pyrenew.observation.types import ObservationSample

from pyrenew.observation.base import BaseObservationProcess
from pyrenew.observation.count_observations import Counts, CountsBySubpop
from pyrenew.observation.measurements import Measurements
from pyrenew.observation.negativebinomial import NegativeBinomialObservation
from pyrenew.observation.noise import (
    CountNoise,
    HierarchicalNormalNoise,
    MeasurementNoise,
    NegativeBinomialNoise,
    PoissonNoise,
)
from pyrenew.observation.poisson import PoissonObservation

__all__ = [
    # Existing (kept for backward compatibility)
    "NegativeBinomialObservation",
    "PoissonObservation",
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
    "Counts",
    "CountsBySubpop",
    "Measurements",
]
