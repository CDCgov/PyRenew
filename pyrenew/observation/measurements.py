# numpydoc ignore=GL08
"""
Continuous measurement observation processes.

Abstract base for any population-level continuous measurements (wastewater,
air quality, serology, etc.) with signal-specific processing.
"""

from typing import Any

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
    name
        Unique name for this observation process. Used to prefix all
        numpyro sample and deterministic site names.
    temporal_pmf_rv
        Temporal distribution PMF (e.g., shedding kinetics for wastewater).
    noise
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
        name
            Unique name for this observation process. Used to prefix all
            numpyro sample and deterministic site names.
        temporal_pmf_rv
            Temporal distribution PMF (e.g., shedding kinetics).
        noise
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

    def validate_data(
        self,
        n_total: int,
        n_subpops: int,
        times: ArrayLike | None = None,
        subpop_indices: ArrayLike | None = None,
        sensor_indices: ArrayLike | None = None,
        n_sensors: int | None = None,
        obs: ArrayLike | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Validate measurement observation data.

        Parameters
        ----------
        n_total
            Total number of time steps (n_init + n_days_post_init).
        n_subpops
            Number of subpopulations.
        times
            Day index for each observation on the shared time axis.
        subpop_indices
            Subpopulation index for each observation (0-indexed).
        sensor_indices
            Sensor index for each observation (0-indexed).
        n_sensors
            Total number of measurement sensors.
        obs
            Observed measurements (n_obs,).
        **kwargs
            Additional keyword arguments (ignored).

        Raises
        ------
        ValueError
            If times, subpop_indices, or sensor_indices are out of bounds,
            or if obs and times have mismatched lengths.
        """
        if times is not None:
            self._validate_times(times, n_total)
            if obs is not None:
                self._validate_obs_times_shape(obs, times)
        if subpop_indices is not None:
            self._validate_subpop_indices(subpop_indices, n_subpops)
        if sensor_indices is not None and n_sensors is not None:
            self._validate_index_array(sensor_indices, n_sensors, "sensor_indices")

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
        infections
            Infections from the infection process.
            Shape: (n_total, n_subpops)
        times
            Day index for each observation on the shared time axis.
            Must be in range [0, n_total). Shape: (n_obs,)
        subpop_indices
            Subpopulation index for each observation (0-indexed).
            Shape: (n_obs,)
        sensor_indices
            Sensor index for each observation (0-indexed).
            Shape: (n_obs,)
        n_sensors
            Total number of measurement sensors.
        obs
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
