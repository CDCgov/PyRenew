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


class Measurements(BaseObservationProcess):
    """
    Abstract base for continuous measurement observations.

    Subclasses implement signal-specific transformations from infections
    to expected measurement values, then add measurement noise.

    Parameters
    ----------
    temporal_pmf_rv : RandomVariable
        Temporal distribution PMF (e.g., shedding kinetics for wastewater).
    noise : MeasurementNoise
        Noise model for continuous measurements
        (e.g., HierarchicalNormalNoise).

    Notes
    -----
    Subclasses must implement ``_expected_signal()`` according to their
    specific signal processing (e.g., wastewater shedding kinetics,
    dilution factors, etc.).

    See Also
    --------
    pyrenew.observation.noise.HierarchicalNormalNoise :
        Suitable noise model for site-level measurements
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
            Noise model (e.g., HierarchicalNormalNoise with site effects).
        """
        super().__init__(temporal_pmf_rv=temporal_pmf_rv)
        self.noise = noise

    def infection_resolution(self) -> str:
        """
        Return "site" for measurement observations.

        Measurement observations require site-level infections
        because each measurement site corresponds to a specific catchment.

        Returns
        -------
        str
            ``"site"``
        """
        return "site"

    def sample(
        self,
        infections: ArrayLike,
        subpop_indices: ArrayLike,
        site_indices: ArrayLike,
        times: ArrayLike,
        concentrations: ArrayLike | None,
        n_sites: int,
    ) -> ArrayLike:
        """
        Sample measurements from observed sites.

        This method does not perform runtime validation of index values
        (times, subpop_indices, site_indices). Validate observation data
        before sampling.

        Transforms infections to expected values via signal-specific processing
        (``_expected_signal``), then applies noise model.

        Parameters
        ----------
        infections : ArrayLike
            Infections from the infection process.
            Shape: (n_days, n_sites)
        subpop_indices : ArrayLike
            Subpopulation index for each observation (0-indexed).
            Shape: (n_obs,)
        site_indices : ArrayLike
            Site index for each observation (0-indexed).
            Shape: (n_obs,)
        times : ArrayLike
            Day index for each observation (0-indexed).
            Shape: (n_obs,)
        concentrations : ArrayLike | None
            Observed measurements (n_obs,), or None for prior sampling.
        n_sites : int
            Total number of measurement sites.

        Returns
        -------
        ArrayLike
            Observed or sampled measurements.
            Shape: (n_obs,)
        """
        expected_values = self._expected_signal(infections)

        self._deterministic("expected_log_conc", expected_values)

        expected_obs = expected_values[times, subpop_indices]

        return self.noise.sample(
            name="concentrations",
            expected=expected_obs,
            obs=concentrations,
            site_indices=site_indices,
            n_sites=n_sites,
        )
