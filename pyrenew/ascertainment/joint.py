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


class JointAscertainment(AscertainmentModel):
    """
    Joint prior for scalar ascertainment rates across multiple signals.

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

    Each returned rate is scalar and constant over the model time axis.
    """

    def __init__(
        self,
        name: str,
        signals: tuple[str, ...],
        baseline_rates: ArrayLike,
        scale_tril: ArrayLike | None = None,
        covariance_matrix: ArrayLike | None = None,
        precision_matrix: ArrayLike | None = None,
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
        """
        super().__init__(name=name, signals=signals)
        self.baseline_rates: Array = jnp.asarray(baseline_rates)
        self.scale_tril: Array | None = self._optional_array(scale_tril)
        self.covariance_matrix: Array | None = self._optional_array(covariance_matrix)
        self.precision_matrix: Array | None = self._optional_array(precision_matrix)
        self._validate_parameters()
        self.baseline_logits: Array = logit(self.baseline_rates)
        self.distribution: dist.MultivariateNormal = dist.MultivariateNormal(
            loc=self.baseline_logits,
            scale_tril=self.scale_tril,
            covariance_matrix=self.covariance_matrix,
            precision_matrix=self.precision_matrix,
        )

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

    def _validate_parameters(self) -> None:
        """
        Validate constructor parameters.
        """
        n_signals = len(self.signals)
        if self.baseline_rates.shape != (n_signals,):
            raise ValueError(
                "baseline_rates must have shape "
                f"({n_signals},), got shape {self.baseline_rates.shape}."
            )
        if jnp.any(self.baseline_rates <= 0) or jnp.any(self.baseline_rates >= 1):
            raise ValueError(
                "baseline_rates must contain probabilities in (0, 1), "
                f"got {self.baseline_rates}."
            )

    def sample(self, **kwargs: object) -> Mapping[str, ArrayLike]:
        """
        Sample jointly distributed scalar ascertainment rates.

        Parameters
        ----------
        **kwargs
            Additional model-context arguments, ignored.

        Returns
        -------
        Mapping[str, ArrayLike]
            Mapping from signal name to sampled scalar ascertainment rate.
        """
        eta = numpyro.sample(
            f"{self.name}_eta",
            self.distribution,
        )
        rates = expit(eta)

        result = {}
        for signal, rate in zip(self.signals, rates):
            numpyro.deterministic(f"{self.name}_{signal}", rate)
            result[signal] = rate

        return result
