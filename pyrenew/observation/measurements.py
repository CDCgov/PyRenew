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
        temporal_pmf_rv: RandomVariable,
        noise: MeasurementNoise,
    ) -> None:
        """
        Initialize measurement observation base.

        Parameters
        ----------
        temporal_pmf_rv : RandomVariable
            Temporal distribution PMF (e.g., shedding kinetics).
        noise : MeasurementNoise
            Noise model (e.g., HierarchicalNormalNoise with sensor effects).
        """
        super().__init__(temporal_pmf_rv=temporal_pmf_rv)
        self.noise = noise

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"temporal_pmf_rv={self.temporal_pmf_rv!r}, "
            f"noise={self.noise!r})"
        )

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
        subpop_indices: ArrayLike,
        sensor_indices: ArrayLike,
        times: ArrayLike,
        obs: ArrayLike | None,
        n_sensors: int,
    ) -> ObservationSample:
        """
        Sample measurements from observed sensors.

        This method does not perform runtime validation of index values
        (times, subpop_indices, sensor_indices). Validate observation data
        before sampling.

        Transforms infections to predicted values via signal-specific processing
        (``_predicted_obs``), then applies noise model.

        Parameters
        ----------
        infections : ArrayLike
            Infections from the infection process.
            Shape: (n_days, n_subpops)
        subpop_indices : ArrayLike
            Subpopulation index for each observation (0-indexed).
            Shape: (n_obs,)
        sensor_indices : ArrayLike
            Sensor index for each observation (0-indexed).
            Shape: (n_obs,)
        times : ArrayLike
            Day index for each observation (0-indexed).
            Shape: (n_obs,)
        obs : ArrayLike | None
            Observed measurements (n_obs,), or None for prior sampling.
        n_sensors : int
            Total number of measurement sensors.

        Returns
        -------
        ObservationSample
            Named tuple with `observed` (sampled/conditioned measurements) and
            `predicted` (predicted values before noise, shape: n_days x n_subpops).
        """
        predicted_values = self._predicted_obs(infections)

        self._deterministic("predicted_log_conc", predicted_values)

        predicted_obs = predicted_values[times, subpop_indices]

        observed = self.noise.sample(
            name="concentrations",
            predicted=predicted_obs,
            obs=obs,
            sensor_indices=sensor_indices,
            n_sensors=n_sensors,
        )

        return ObservationSample(observed=observed, predicted=predicted_values)
