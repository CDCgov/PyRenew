# numpydoc ignore=GL08
"""
Joint ascertainment models.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.scipy.special import expit, logit
from jax.typing import ArrayLike

from pyrenew.ascertainment.base import AscertainmentModel
from pyrenew.latent import TemporalProcess


class JointAscertainment(AscertainmentModel):
    """Joint prior for baseline ascertainment rates across multiple signals.

    This model is useful when multiple observation streams have distinct but
    related probabilities of observing latent incidence. For example, hospital
    admissions and emergency department visits may have different
    infection-to-observation ratios, while still being correlated because both
    depend on care-seeking behavior, testing practices, or reporting systems.

    The model samples one logit multivariate normal vector given natural-scale
    baseline ascertainment rates.

    ```text
    eta ~ MultivariateNormal(logit(baseline_rates), covariance)
    ascertainment_rate_j = sigmoid(eta_j)
    ```

    Each sampled baseline is scalar. Optional temporal processes can vary any
    subset of the final rates over the model time axis.
    """

    def __init__(
        self,
        name: str,
        signals: tuple[str, ...],
        baseline_rates: ArrayLike,
        scale_tril: ArrayLike | None = None,
        covariance_matrix: ArrayLike | None = None,
        precision_matrix: ArrayLike | None = None,
        temporal_processes: Mapping[str, TemporalProcess] | None = None,
    ) -> None:
        """
        Initialize a joint scalar ascertainment model.

        Parameters
        ----------
        name
            Name of the ascertainment model.
        signals
            Unique signal names, such as ``("hospital", "ed_visits")``. The
            order corresponds to entries in ``baseline_rates`` and the covariance
            parameter.
        baseline_rates
            Natural-scale baseline ascertainment rates. Shape ``(n_signals,)``.
            Values must be probabilities in ``(0, 1)``. A value of ``0.01``
            centers the corresponding ascertainment rate near 1 percent before
            accounting for covariance.
        scale_tril
            Lower-triangular scale matrix for the multivariate normal on the
            logit scale. Exactly one covariance parameter must be supplied.
        covariance_matrix
            Covariance matrix for the multivariate normal on the logit scale.
            Exactly one covariance parameter must be supplied.
        precision_matrix
            Precision matrix for the multivariate normal on the logit scale.
            Exactly one covariance parameter must be supplied.
        temporal_processes
            Optional temporal processes keyed by signal name. Signals without
            a process retain their scalar baseline rate.
        """
        super().__init__(
            name=name,
            signals=signals,
            temporal_processes=temporal_processes,
        )
        baseline_rates_array = jnp.asarray(baseline_rates)
        scale_tril_array = self._optional_array(scale_tril)
        covariance_matrix_array = self._optional_array(covariance_matrix)
        precision_matrix_array = self._optional_array(precision_matrix)
        self._validate_parameters(baseline_rates_array)
        self.distribution: dist.MultivariateNormal = dist.MultivariateNormal(
            loc=logit(baseline_rates_array),
            scale_tril=scale_tril_array,
            covariance_matrix=covariance_matrix_array,
            precision_matrix=precision_matrix_array,
        )

    @property
    def baseline_rates(self) -> Array:
        """
        Natural-scale baseline ascertainment rates.

        Returns
        -------
        Array
            Distribution location transformed from logit scale to probability
            scale.
        """
        return expit(self.distribution.loc)

    @staticmethod
    def _optional_array(value: ArrayLike | None) -> Array | None:
        """
        Convert optional array-like values to JAX arrays.

        Returns
        -------
        Array | None
            ``None`` if ``value`` is ``None``; otherwise ``value`` converted
            to a JAX array.
        """
        if value is None:
            return None
        return jnp.asarray(value)

    def _validate_parameters(self, baseline_rates: Array) -> None:
        """
        Validate constructor parameters.
        """
        n_signals = len(self.signals)
        if baseline_rates.shape != (n_signals,):
            raise ValueError(
                "baseline_rates must have shape "
                f"({n_signals},), got shape {baseline_rates.shape}."
            )
        if jnp.any(baseline_rates <= 0) or jnp.any(baseline_rates >= 1):
            raise ValueError(
                "baseline_rates must contain probabilities in (0, 1), "
                f"got {baseline_rates}."
            )

    def _sample_baseline_rates(self) -> Mapping[str, ArrayLike]:
        """Sample jointly distributed scalar baseline rates.

        Returns
        -------
        Mapping[str, ArrayLike]
            Sampled scalar baseline rates keyed by signal name.
        """
        eta = numpyro.sample(
            f"{self.name}_eta",
            self.distribution,
        )
        rates = expit(eta)

        result: dict[str, ArrayLike] = {}
        for signal, rate in zip(self.signals, rates):
            result[signal] = rate

        return result
