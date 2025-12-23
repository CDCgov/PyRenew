# numpydoc ignore=GL08
"""
Observation processes for connecting infections to observed data.

Architecture
------------
``BaseObservationProcess`` is the abstract base. Concrete subclasses:

- ``Counts``: Jurisdiction-level counts (admissions, deaths)
- ``CountsBySite``: Site-specific disaggregated counts
- ``Measurements``: Continuous site-level signals (e.g., wastewater)

All observation processes implement:

- ``_expected_signal(infections)``: transforms infections to expected values
- ``sample()``: calls ``_expected_signal()`` then applies noise model

Noise models (``CountNoise``, ``MeasurementNoise``) are composable—pass them
to observation constructors to control the output distribution.
"""

from pyrenew.observation.base import BaseObservationProcess
from pyrenew.observation.count_observations import Counts, CountsBySite
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
    # New base classes
    "BaseObservationProcess",
    # New noise models
    "CountNoise",
    "PoissonNoise",
    "NegativeBinomialNoise",
    "MeasurementNoise",
    "HierarchicalNormalNoise",
    # New observation processes
    "Counts",
    "CountsBySite",
    "Measurements",
]
