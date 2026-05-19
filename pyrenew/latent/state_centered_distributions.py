"""NumPyro distributions for state-centered temporal-process priors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.continuous import Normal
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key


class StateRandomWalk(Distribution):
    r"""
    State-centered random-walk prior on a post-initial state path.

    Given a deterministic initial state $x_0$ = ``initial_loc``:

    $$
    x_t \sim \mathrm{Normal}(x_{t-1}, \sigma), \quad t = 1, \dots, T
    $$

    The sampled value is the post-initial path
    $[x_1, x_2, \ldots, x_{\mathrm{num\_steps}}]$ of length ``num_steps``.
    """

    arg_constraints = {
        "scale": constraints.positive,
        "initial_loc": constraints.real,
    }
    support = constraints.real_vector
    reparametrized_params = ["scale", "initial_loc"]
    pytree_aux_fields = ("num_steps",)

    def __init__(
        self,
        scale: ArrayLike,
        initial_loc: ArrayLike = 0.0,
        num_steps: int = 1,
        *,
        validate_args: bool | None = None,
    ) -> None:
        """Construct a state-centered random-walk distribution."""
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError(f"num_steps must be a positive integer; got {num_steps!r}")
        self.scale = scale
        self.initial_loc = initial_loc
        self.num_steps = num_steps

        batch_shape = lax.broadcast_shapes(
            jnp.shape(scale),
            jnp.shape(initial_loc),
        )
        super().__init__(batch_shape, (num_steps,), validate_args=validate_args)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        """
        Forward-sample a post-initial random-walk state path.

        Returns
        -------
        ArrayLike
            Array of shape ``sample_shape + batch_shape + (num_steps,)``.
        """
        assert is_prng_key(key)

        per_step_shape = sample_shape + self.batch_shape
        scale = jnp.broadcast_to(jnp.asarray(self.scale), per_step_shape)
        initial_loc = jnp.broadcast_to(jnp.asarray(self.initial_loc), per_step_shape)
        noise = random.normal(key, shape=per_step_shape + (self.num_steps,))
        increments = scale[..., jnp.newaxis] * noise
        return initial_loc[..., jnp.newaxis] + jnp.cumsum(increments, axis=-1)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """
        Compute the log-density of an observed post-initial state path.

        Parameters
        ----------
        value
            Post-initial path of shape
            ``sample_shape + batch_shape + (num_steps,)``.

        Returns
        -------
        ArrayLike
            Log-density of shape ``sample_shape + batch_shape``.
        """
        scale = jnp.asarray(self.scale)
        initial_loc = jnp.asarray(self.initial_loc)
        init_with_event = jnp.expand_dims(initial_loc, -1)
        init_bcast = jnp.broadcast_to(init_with_event, value.shape[:-1] + (1,))
        v = jnp.concatenate([init_bcast, value], axis=-1)
        step_probs = Normal(v[..., :-1], jnp.expand_dims(scale, -1)).log_prob(
            v[..., 1:]
        )
        return jnp.sum(step_probs, axis=-1)


class StateAR1(Distribution):
    r"""
    State-centered AR(1) prior on a length-``num_steps`` state path.

    Generative form:

    $$
    x_0 \sim \mathrm{Normal}(\mu_0, \sigma_{\text{stat}})
    $$
    $$
    x_t \sim \mathrm{Normal}(\phi \, x_{t-1}, \sigma), \quad t = 1, \dots, T-1
    $$

    where $\sigma_{\text{stat}} = \sigma / \sqrt{1 - \phi^2}$ is the
    stationary standard deviation, $\mu_0$ is ``initial_loc``, $\phi$ is
    ``autoreg``, and $\sigma$ is ``scale``.

    The sampled value is the full path $[x_0, x_1, \ldots, x_{T-1}]$.

    Parameters
    ----------
    autoreg
        AR(1) coefficient $\phi$. For stationarity, $|\phi| < 1$; this is
        not enforced.
    scale
        Innovation standard deviation $\sigma$. Must be positive.
    initial_loc
        Prior mean $\mu_0$ of the initial state $x_0$. Defaults to ``0.0``.
    num_steps
        Length of the state path. Must be a positive integer.
    validate_args
        Forwarded to the base [`numpyro.distributions.Distribution`][].
    """

    arg_constraints = {
        "autoreg": constraints.real,
        "scale": constraints.positive,
        "initial_loc": constraints.real,
    }
    support = constraints.real_vector
    reparametrized_params = ["autoreg", "scale", "initial_loc"]
    pytree_aux_fields = ("num_steps",)

    def __init__(
        self,
        autoreg: ArrayLike,
        scale: ArrayLike,
        initial_loc: ArrayLike = 0.0,
        num_steps: int = 1,
        *,
        validate_args: bool | None = None,
    ) -> None:
        """
        Construct a state-centered AR(1) distribution.

        Raises
        ------
        ValueError
            If ``num_steps`` is not a positive integer.
        """
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError(f"num_steps must be a positive integer; got {num_steps!r}")
        self.autoreg = autoreg
        self.scale = scale
        self.initial_loc = initial_loc
        self.num_steps = num_steps

        batch_shape = lax.broadcast_shapes(
            jnp.shape(autoreg),
            jnp.shape(scale),
            jnp.shape(initial_loc),
        )
        event_shape = (num_steps,)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        """
        Forward-sample a state path.

        Returns
        -------
        ArrayLike
            Array of shape ``sample_shape + batch_shape + (num_steps,)``.
        """
        assert is_prng_key(key)

        per_step_shape = sample_shape + self.batch_shape
        autoreg = jnp.broadcast_to(jnp.asarray(self.autoreg), per_step_shape)
        scale = jnp.broadcast_to(jnp.asarray(self.scale), per_step_shape)
        initial_loc = jnp.broadcast_to(jnp.asarray(self.initial_loc), per_step_shape)
        stationary_sd = scale / jnp.sqrt(1 - autoreg**2)

        noise = random.normal(key, shape=(self.num_steps,) + per_step_shape)
        z0 = noise[0]
        x0 = initial_loc + stationary_sd * z0

        if self.num_steps == 1:
            return x0[..., jnp.newaxis]

        def step(
            prev: ArrayLike, z_t: ArrayLike
        ) -> tuple[ArrayLike, ArrayLike]:  # numpydoc ignore=GL08
            new = autoreg * prev + scale * z_t
            return new, new

        _, xs = lax.scan(step, x0, noise[1:])
        path_time_first = jnp.concatenate([x0[jnp.newaxis], xs], axis=0)
        return jnp.moveaxis(path_time_first, 0, -1)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """
        Compute the log-density of an observed state path.

        Parameters
        ----------
        value
            State path of shape ``sample_shape + batch_shape + (num_steps,)``.

        Returns
        -------
        ArrayLike
            Log-density of shape ``sample_shape + batch_shape``.
        """
        scale = jnp.asarray(self.scale)
        autoreg = jnp.asarray(self.autoreg)
        stationary_sd = scale / jnp.sqrt(1 - autoreg**2)

        init_prob = Normal(self.initial_loc, stationary_sd).log_prob(value[..., 0])

        scale_t = jnp.expand_dims(scale, -1)
        autoreg_t = jnp.expand_dims(autoreg, -1)
        step_locs = autoreg_t * value[..., :-1]
        step_probs = Normal(step_locs, scale_t).log_prob(value[..., 1:])
        return init_prob + jnp.sum(step_probs, axis=-1)


class StateDifferencedAR1(Distribution):
    r"""
    State-centered differenced AR(1) prior on a length-``num_steps`` post-initial path.

    Generative form, given a deterministic initial state $x_0$ = ``initial_loc``:

    $$
    x_1 \sim \mathrm{Normal}(x_0, \sigma_{\text{stat}})
    $$
    $$
    x_t \sim \mathrm{Normal}(x_{t-1} + \phi \, (x_{t-1} - x_{t-2}), \sigma),
    \quad t \geq 2
    $$

    where $\sigma_{\text{stat}} = \sigma / \sqrt{1 - \phi^2}$, $\phi$ is
    ``autoreg``, and $\sigma$ is ``scale``.

    The sampled value is the post-initial path
    $[x_1, x_2, \ldots, x_{\mathrm{num\_steps}}]$ of length ``num_steps``.
    The initial state $x_0$ is not part of the sample; it is supplied as
    ``initial_loc`` and used to score the first transition.

    Parameters
    ----------
    autoreg
        AR(1) coefficient $\phi$ on first differences. For stationarity,
        $|\phi| < 1$; this is not enforced.
    scale
        Innovation standard deviation $\sigma$. Must be positive.
    initial_loc
        Deterministic initial state $x_0$. Used to score the first
        transition; not itself sampled.
    num_steps
        Length of the post-initial path. Must be a positive integer.
    validate_args
        Forwarded to the base [`numpyro.distributions.Distribution`][].
    """

    arg_constraints = {
        "autoreg": constraints.real,
        "scale": constraints.positive,
        "initial_loc": constraints.real,
    }
    support = constraints.real_vector
    reparametrized_params = ["autoreg", "scale", "initial_loc"]
    pytree_aux_fields = ("num_steps",)

    def __init__(
        self,
        autoreg: ArrayLike,
        scale: ArrayLike,
        initial_loc: ArrayLike = 0.0,
        num_steps: int = 1,
        *,
        validate_args: bool | None = None,
    ) -> None:
        """
        Construct a state-centered differenced AR(1) distribution.

        Raises
        ------
        ValueError
            If ``num_steps`` is not a positive integer.
        """
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError(f"num_steps must be a positive integer; got {num_steps!r}")
        self.autoreg = autoreg
        self.scale = scale
        self.initial_loc = initial_loc
        self.num_steps = num_steps

        batch_shape = lax.broadcast_shapes(
            jnp.shape(autoreg),
            jnp.shape(scale),
            jnp.shape(initial_loc),
        )
        event_shape = (num_steps,)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        """
        Forward-sample a post-initial path.

        Returns
        -------
        ArrayLike
            Array of shape ``sample_shape + batch_shape + (num_steps,)``.
        """
        assert is_prng_key(key)

        per_step_shape = sample_shape + self.batch_shape
        autoreg = jnp.broadcast_to(jnp.asarray(self.autoreg), per_step_shape)
        scale = jnp.broadcast_to(jnp.asarray(self.scale), per_step_shape)
        initial_loc = jnp.broadcast_to(jnp.asarray(self.initial_loc), per_step_shape)
        stationary_sd = scale / jnp.sqrt(1 - autoreg**2)

        noise = random.normal(key, shape=(self.num_steps,) + per_step_shape)
        z1 = noise[0]
        x1 = initial_loc + stationary_sd * z1

        if self.num_steps == 1:
            return x1[..., jnp.newaxis]

        def step(
            carry: tuple[ArrayLike, ArrayLike], z_t: ArrayLike
        ) -> tuple[tuple[ArrayLike, ArrayLike], ArrayLike]:  # numpydoc ignore=GL08
            prev_2, prev_1 = carry
            new = prev_1 + autoreg * (prev_1 - prev_2) + scale * z_t
            return (prev_1, new), new

        _, xs = lax.scan(step, (initial_loc, x1), noise[1:])
        path_time_first = jnp.concatenate([x1[jnp.newaxis], xs], axis=0)
        return jnp.moveaxis(path_time_first, 0, -1)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """
        Compute the log-density of an observed post-initial path.

        Parameters
        ----------
        value
            Post-initial path of shape
            ``sample_shape + batch_shape + (num_steps,)``.

        Returns
        -------
        ArrayLike
            Log-density of shape ``sample_shape + batch_shape``.
        """
        scale = jnp.asarray(self.scale)
        autoreg = jnp.asarray(self.autoreg)
        initial_loc = jnp.asarray(self.initial_loc)
        stationary_sd = scale / jnp.sqrt(1 - autoreg**2)

        init_prob = Normal(initial_loc, stationary_sd).log_prob(value[..., 0])

        init_with_event = jnp.expand_dims(initial_loc, -1)
        init_bcast = jnp.broadcast_to(init_with_event, value.shape[:-1] + (1,))
        v = jnp.concatenate([init_bcast, value], axis=-1)

        prev_delta = v[..., 1:-1] - v[..., :-2]
        scale_t = jnp.expand_dims(scale, -1)
        autoreg_t = jnp.expand_dims(autoreg, -1)
        means = v[..., 1:-1] + autoreg_t * prev_delta
        step_probs = Normal(means, scale_t).log_prob(v[..., 2:])
        return init_prob + jnp.sum(step_probs, axis=-1)
