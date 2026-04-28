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


class JointAscertainment(AscertainmentModel):
    """
    Joint logit-normal prior over signal-specific ascertainment rates.

    Samples one vector-valued latent parameter on the logit scale and maps each
    component to ``(0, 1)`` with the sigmoid function.
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
        Initialize a joint ascertainment model.

        Parameters
        ----------
        name
            Name of the ascertainment model.
        signals
            Unique signal names. Order corresponds to entries in ``loc`` and
            the multivariate normal covariance parameter.
        loc
            Mean vector on the logit scale. Shape ``(n_signals,)``.
        scale_tril
            Lower-triangular scale matrix for the multivariate normal.
        covariance_matrix
            Covariance matrix for the multivariate normal.
        precision_matrix
            Precision matrix for the multivariate normal.
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
                "loc must have shape "
                f"({n_signals},), got shape {self.loc.shape}."
            )

        covariance_params = (
            self.scale_tril,
            self.covariance_matrix,
            self.precision_matrix,
        )
        n_covariance_params = sum(param is not None for param in covariance_params)
        if n_covariance_params != 1:
            raise ValueError(
                "Exactly one of scale_tril, covariance_matrix, or "
                "precision_matrix must be provided."
            )

        matrix_shape = (n_signals, n_signals)
        for param_name, param in (
            ("scale_tril", self.scale_tril),
            ("covariance_matrix", self.covariance_matrix),
            ("precision_matrix", self.precision_matrix),
        ):
            if param is not None and param.shape != matrix_shape:
                raise ValueError(
                    f"{param_name} must have shape {matrix_shape}, "
                    f"got shape {param.shape}."
                )

    def _distribution(self) -> dist.MultivariateNormal:
        """
        Construct the joint latent distribution.
        """
        if self.scale_tril is not None:
            return dist.MultivariateNormal(
                loc=self.loc,
                scale_tril=self.scale_tril,
            )
        if self.covariance_matrix is not None:
            return dist.MultivariateNormal(
                loc=self.loc,
                covariance_matrix=self.covariance_matrix,
            )
        return dist.MultivariateNormal(
            loc=self.loc,
            precision_matrix=self.precision_matrix,
        )

    def sample(self, **kwargs: object) -> Mapping[str, ArrayLike]:
        """
        Sample jointly distributed ascertainment rates.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments, ignored.

        Returns
        -------
        Mapping[str, ArrayLike]
            Mapping from signal name to sampled ascertainment rate.
        """
        eta = numpyro.sample(
            f"{self.name}_eta",
            self._distribution(),
        )
        rates = jnn.sigmoid(eta)

        result = {}
        for i, signal in enumerate(self.signals):
            rate = rates[i]
            numpyro.deterministic(f"{self.name}_{signal}", rate)
            result[signal] = rate

        return result
