# numpydoc ignore=GL08

import jax
import jax.numpy as jnp
import numpyro
import numpyro.util
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes, validate_sample


class CensoredNormal(numpyro.distributions.Distribution):
    """
    Censored normal distribution under which samples
    are truncated to lie within a specified interval.
    This implementation is adapted from
    https://github.com/dylanhmorris/host-viral-determinants/blob/main/src/distributions.py
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    pytree_data_fields = (
        "loc",
        "scale",
        "lower_limit",
        "upper_limit",
        "_support",
    )

    def __init__(
        self,
        loc: float = 0,
        scale: float = 1,
        lower_limit: float = -jnp.inf,
        upper_limit: float = jnp.inf,
        validate_args: bool | None = None,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        loc
            The mean of the normal distribution.
            Defaults to 0.
        scale
            The standard deviation of the normal
            distribution. Must be positive. Defaults to 1.
        lower_limit
            The lower bound of the interval for censoring.
            Defaults to -inf (no lower bound).
        upper_limit
            The upper bound of the interval for censoring.
            Defaults to inf (no upper bound).
        validate_args
            If True, checks if parameters are valid.
            Defaults to None.

        Returns
        -------
        None
        """
        self.loc, self.scale = promote_shapes(loc, scale)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self._support = constraints.interval(self.lower_limit, self.upper_limit)
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.normal_ = numpyro.distributions.Normal(
            loc=loc, scale=scale, validate_args=validate_args
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:  # numpydoc ignore=GL08
        return self._support

    def sample(self, key: ArrayLike, sample_shape: tuple = ()) -> ArrayLike:
        """
        Generates samples from the censored normal distribution.

        Returns
        -------
        Array
            Containing samples from the censored normal distribution.
        """
        assert numpyro.util.is_prng_key(key)
        result = self.normal_.sample(key, sample_shape)
        return jnp.clip(result, min=self.lower_limit, max=self.upper_limit)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """
        Computes the log probability density of a given value(s) under
        the censored normal distribution.

        Returns
        -------
        Array
            Containing log probability of the given value(s)
            under the censored normal distribution
        """
        rescaled_ulim = (self.upper_limit - self.loc) / self.scale
        rescaled_llim = (self.lower_limit - self.loc) / self.scale
        lim_val = jnp.where(
            value <= self.lower_limit,
            jax.scipy.special.log_ndtr(rescaled_llim),
            jax.scipy.special.log_ndtr(-rescaled_ulim),
        )
        # we exploit the fact that for the
        # standard normal, P(x > a) = P(-x < a)
        # to compute the log complementary CDF
        inbounds = jnp.logical_and(value > self.lower_limit, value < self.upper_limit)
        result = jnp.where(inbounds, self.normal_.log_prob(value), lim_val)

        return result
