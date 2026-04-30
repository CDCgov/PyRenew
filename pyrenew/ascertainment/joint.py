# numpydoc ignore=GL08
"""
Joint ascertainment models.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.nn as jnn
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from pyrenew.ascertainment.base import AscertainmentModel

_COVARIANCE_PARAMETER_NAMES = (
    "scale_tril",
    "covariance_matrix",
    "precision_matrix",
)


class JointAscertainment(AscertainmentModel):
    """
    Joint prior for scalar ascertainment rates across multiple signals.

    This model is useful when multiple observation streams have distinct but
    related probabilities of observing latent incidence. For example, hospital
    admissions and emergency department visits may have different
    infection-to-observation ratios, while still being correlated because both
    depend on care-seeking behavior, testing practices, or reporting systems.

    The model samples one multivariate normal vector on the logit scale and
    transforms each component to a probability:

    ```text
    eta ~ MultivariateNormal(loc, covariance)
    ascertainment_rate_j = sigmoid(eta_j)
    ```

    Each returned rate is scalar and constant over the model time axis. Use
    ``TimeVaryingAscertainment`` when the probability of observing incidence
    should vary through time.
    """

    def __init__(
        self,
        name: str,
        signals: tuple[str, ...],
        loc: ArrayLike,
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
            order corresponds to entries in ``loc`` and the covariance
            parameter.
        loc
            Mean vector on the logit scale. Shape ``(n_signals,)``. A value of
            ``logit(0.01)`` centers the corresponding ascertainment rate near
            1 percent before accounting for covariance.
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
        self.loc = jnp.asarray(loc)
        self.scale_tril = self._optional_array(scale_tril)
        self.covariance_matrix = self._optional_array(covariance_matrix)
        self.precision_matrix = self._optional_array(precision_matrix)
        self._validate_parameters()

    @staticmethod
    def _optional_array(value: ArrayLike | None) -> ArrayLike | None:
        """
        Convert optional array-like values to JAX arrays.

        Returns
        -------
        ArrayLike | None
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
        if self.loc.shape != (n_signals,):
            raise ValueError(
                f"loc must have shape ({n_signals},), got shape {self.loc.shape}."
            )

        n_covariance_params = sum(
            getattr(self, name) is not None for name in _COVARIANCE_PARAMETER_NAMES
        )
        if n_covariance_params != 1:
            raise ValueError(
                "Exactly one of scale_tril, covariance_matrix, or "
                "precision_matrix must be provided."
            )

        matrix_shape = (n_signals, n_signals)
        for name in _COVARIANCE_PARAMETER_NAMES:
            param = getattr(self, name)
            if param is not None and param.shape != matrix_shape:
                raise ValueError(
                    f"{name} must have shape {matrix_shape}, got shape {param.shape}."
                )

    def _distribution(self) -> dist.MultivariateNormal:
        """
        Construct the joint latent distribution.

        Returns
        -------
        numpyro.distributions.MultivariateNormal
            Joint latent distribution on the logit scale.
        """
        for name in _COVARIANCE_PARAMETER_NAMES:
            value = getattr(self, name)
            if value is not None:
                return dist.MultivariateNormal(loc=self.loc, **{name: value})

        raise ValueError(
            "Exactly one of scale_tril, covariance_matrix, or "
            "precision_matrix must be provided."
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
            self._distribution(),
        )
        rates = jnn.sigmoid(eta)

        result = {}
        for signal, rate in zip(self.signals, rates):
            numpyro.deterministic(f"{self.name}_{signal}", rate)
            result[signal] = rate

        return result
