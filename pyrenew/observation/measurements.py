# numpydoc ignore=GL08
"""
Continuous measurement observation processes.

Abstract base for any population-level continuous measurements (wastewater,
air quality, serology, etc.) with signal-specific processing.
"""

from jax.typing import ArrayLike

from pyrenew.metaclass import RandomVariable
from pyrenew.observation.base import BaseObservationProcess
from pyrenew.observation.noise import MeasurementNoise
from pyrenew.observation.types import ObservationSample


class Measurements(BaseObservationProcess):
    """
    Abstract base for continuous measurement observations.

    Subclasses implement signal-specific transformations from infections
    to predicted measurement values, then add measurement noise.

    Parameters
    ----------
    name : str
        Unique name for this observation process. Used to prefix all
        numpyro sample and deterministic site names.
    temporal_pmf_rv : RandomVariable
        Temporal distribution PMF (e.g., shedding kinetics for wastewater).
    noise : MeasurementNoise
        Noise model for continuous measurements
        (e.g., HierarchicalNormalNoise).

    Notes
    -----
    Subclasses must implement ``_predicted_obs()`` according to their
    specific signal processing (e.g., wastewater shedding kinetics,
    dilution factors, etc.).

    See Also
    --------
    pyrenew.observation.noise.HierarchicalNormalNoise :
        Suitable noise model for sensor-level measurements
    pyrenew.observation.base.BaseObservationProcess :
        Parent class with common observation utilities
    """

    def __init__(
        self,
        name: str,
        temporal_pmf_rv: RandomVariable,
        noise: MeasurementNoise,
    ) -> None:
        """
        Initialize measurement observation base.

        Parameters
        ----------
        name : str
            Unique name for this observation process. Used to prefix all
            numpyro sample and deterministic site names.
        temporal_pmf_rv : RandomVariable
            Temporal distribution PMF (e.g., shedding kinetics).
        noise : MeasurementNoise
            Noise model (e.g., HierarchicalNormalNoise with sensor effects).
        """
        super().__init__(name=name, temporal_pmf_rv=temporal_pmf_rv)
        self.noise = noise

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"temporal_pmf_rv={self.temporal_pmf_rv!r}, "
            f"noise={self.noise!r})"
        )

    def lookback_days(self) -> int:
        """
        Return required lookback days for this observation.

        Temporal PMFs are 0-indexed (effect can occur on day 0), so a PMF
        of length L covers lags 0 to L-1, requiring L-1 initialization points.

        Returns
        -------
        int
            Length of temporal PMF minus 1.
        """
        return len(self.temporal_pmf_rv()) - 1

    def infection_resolution(self) -> str:
        """
        Return "subpop" for measurement observations.

        Measurement observations require subpopulation-level infections
        because each measurement corresponds to a specific catchment area.

        Returns
        -------
        str
            ``"subpop"``
        """
        return "subpop"

    def sample(
        self,
        infections: ArrayLike,
        times: ArrayLike,
        subpop_indices: ArrayLike,
        sensor_indices: ArrayLike,
        n_sensors: int,
        obs: ArrayLike | None = None,
    ) -> ObservationSample:
        """
        Sample measurements from observed sensors.

        Times are on the shared time axis [0, n_total) where
        n_total = n_init + n_days. This method performs direct indexing
        without any offset adjustment.

        Transforms infections to predicted values via signal-specific processing
        (``_predicted_obs``), then applies noise model.

        Parameters
        ----------
        infections : ArrayLike
            Infections from the infection process.
            Shape: (n_total, n_subpops)
        times : ArrayLike
            Day index for each observation on the shared time axis.
            Must be in range [0, n_total). Shape: (n_obs,)
        subpop_indices : ArrayLike
            Subpopulation index for each observation (0-indexed).
            Shape: (n_obs,)
        sensor_indices : ArrayLike
            Sensor index for each observation (0-indexed).
            Shape: (n_obs,)
        n_sensors : int
            Total number of measurement sensors.
        obs : ArrayLike | None
            Observed measurements (n_obs,), or None for prior sampling.

        Returns
        -------
        ObservationSample
            Named tuple with `observed` (sampled/conditioned measurements) and
            `predicted` (predicted values before noise, shape: n_total x n_subpops).
        """
        predicted_values = self._predicted_obs(infections)
        self._deterministic("predicted", predicted_values)

        # Direct indexing on shared time axis - no offset needed
        predicted_obs = predicted_values[times, subpop_indices]

        observed = self.noise.sample(
            name=self._sample_site_name("obs"),
            predicted=predicted_obs,
            obs=obs,
            sensor_indices=sensor_indices,
            n_sensors=n_sensors,
        )

        return ObservationSample(observed=observed, predicted=predicted_values)
