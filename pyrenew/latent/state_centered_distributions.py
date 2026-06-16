"""NumPyro distributions for state-centered temporal-process priors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.continuous import GaussianRandomWalk, Normal
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key


class StateRandomWalk(Distribution):
    r"""
    State-centered random walk on a post-initial state path.

    Given a deterministic initial state $x_0$ = ``initial_loc``:

    $$
    x_t \sim \mathrm{Normal}(x_{t-1}, \sigma), \quad t = 1, \dots, T
    $$

    The sampled value is the post-initial path
    $[x_1, x_2, \ldots, x_{\mathrm{num\_steps}}]$ of length ``num_steps``.

    This is a location shift of [numpyro.distributions.continuous.GaussianRandomWalk][].
    A zero-mean Gaussian random walk supplies the path, and ``initial_loc``
    offsets it so the walk is centered on the initial state rather than zero.
    """

    arg_constraints = {
        "scale": constraints.positive,
        "initial_loc": constraints.real,
    }
    support = constraints.real_vector
    reparametrized_params = ["scale", "initial_loc"]
    pytree_data_fields = ("gaussian_random_walk_", "initial_loc")

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
        self.initial_loc = initial_loc
        self.gaussian_random_walk_ = GaussianRandomWalk(scale, num_steps)

        batch_shape = lax.broadcast_shapes(
            jnp.shape(scale),
            jnp.shape(initial_loc),
        )
        super().__init__(batch_shape, (num_steps,), validate_args=validate_args)

    @property
    def scale(self) -> ArrayLike:
        """
        Innovation standard deviation of the underlying random walk.

        Returns
        -------
        ArrayLike
            The innovation standard deviation $\\sigma$.
        """
        return self.gaussian_random_walk_.scale

    @property
    def num_steps(self) -> int:
        """
        Length of the post-initial state path.

        Returns
        -------
        int
            The number of post-initial steps.
        """
        return self.gaussian_random_walk_.num_steps

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        """
        Forward-sample a post-initial random-walk state path.

        Returns
        -------
        ArrayLike
            Array of shape ``sample_shape + batch_shape + (num_steps,)``.
        """
        assert is_prng_key(key)
        walk = self.gaussian_random_walk_.sample(key, sample_shape)
        # initial_loc carries batch_shape and walk's last axis is time
        # (num_steps); add a trailing length-1 time axis so initial_loc aligns
        # on batch rather than right-aligning against the time axis.
        return jnp.expand_dims(jnp.asarray(self.initial_loc), -1) + walk

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
        # initial_loc carries batch_shape and value's last axis is time
        # (num_steps); add a trailing length-1 time axis so the offset aligns
        # on batch rather than right-aligning against the time axis.
        offset = jnp.expand_dims(jnp.asarray(self.initial_loc), -1)
        return self.gaussian_random_walk_.log_prob(value - offset)


class StateAR1(Distribution):
    r"""
    State-centered AR(1) process for a post-initial state path.

    Generative form, given a deterministic initial state $x_0$ = ``initial_loc``:

    $$
    x_t \sim \mathrm{Normal}(\phi \, x_{t-1}, \sigma), \quad t = 1, \dots, T
    $$

    where $\phi$ is ``autoreg`` and $\sigma$ is ``scale``.

    The sampled value is the post-initial path
    $[x_1, x_2, \ldots, x_{\mathrm{num\_steps}}]$ of length ``num_steps``.
    The initial state $x_0$ is not part of the sample; it is supplied as
    ``initial_loc`` and used to score the first transition. A random initial
    state drawn from the stationary distribution can be handled by the calling
    temporal process as a separate sample site, matching the innovation
    parameterization.

    Parameters
    ----------
    autoreg
        AR(1) coefficient $\phi$. For stationarity, $|\phi| < 1$; this is
        not enforced.
    scale
        Innovation standard deviation $\sigma$. Must be positive.
    initial_loc
        Deterministic initial state $x_0$. Used to score the first
        transition; not itself sampled. Defaults to ``0.0``.
    num_steps
        Length of the post-initial path. Must be a positive integer.
    validate_args
        Forwarded to the base [numpyro.distributions.distribution.Distribution][].
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
        super().__init__(batch_shape, (num_steps,), validate_args=validate_args)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        """
        Forward-sample a post-initial state path.

        Returns
        -------
        ArrayLike
            Array of shape ``sample_shape + batch_shape + (num_steps,)``.
        """
        assert is_prng_key(key)

        per_step_shape = sample_shape + self.batch_shape
        autoreg = jnp.asarray(self.autoreg)
        scale = jnp.asarray(self.scale)
        # lax.scan requires a shape-invariant carry; broadcast the initial
        # state to per_step_shape so it matches the per-step output shape.
        initial_loc = jnp.broadcast_to(jnp.asarray(self.initial_loc), per_step_shape)

        noise = random.normal(key, shape=(self.num_steps,) + per_step_shape)

        def step(
            prev: ArrayLike, z_t: ArrayLike
        ) -> tuple[ArrayLike, ArrayLike]:  # numpydoc ignore=GL08
            new = autoreg * prev + scale * z_t
            return new, new

        _, xs = lax.scan(step, initial_loc, noise)
        return jnp.moveaxis(
            xs, 0, -1
        )  # ensure time is the trailing axis, length num_steps

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
        autoreg = jnp.asarray(self.autoreg)
        initial_loc = jnp.asarray(self.initial_loc)

        per_step_shape = jnp.shape(value)[:-1]
        initial_loc = jnp.broadcast_to(initial_loc, per_step_shape)

        # add length-1 trailing time dimension to fixed-in-time parameters
        initial_loc_t = jnp.expand_dims(initial_loc, -1)
        scale_t = jnp.expand_dims(scale, -1)
        autoreg_t = jnp.expand_dims(autoreg, -1)

        v = jnp.concatenate([initial_loc_t, value], axis=-1)
        step_locs = autoreg_t * v[..., :-1]
        step_probs = Normal(step_locs, scale_t).log_prob(v[..., 1:])
        return jnp.sum(step_probs, axis=-1)


class StateDifferencedAR1(Distribution):
    r"""
    State-centered differenced AR(1) prior on a length-``num_steps`` post-initial path.

    Generative form, given a deterministic initial state $x_0$ = ``initial_loc``
    and initial first difference $d_1$ = ``initial_diff``, so that
    $x_1 = x_0 + d_1$:

    $$
    x_t \sim \mathrm{Normal}(x_{t-1} + \phi \, (x_{t-1} - x_{t-2}), \sigma),
    \quad t \geq 2
    $$

    where $\phi$ is ``autoreg`` and $\sigma$ is ``scale``.

    The sampled value is the post-initial path
    $[x_2, x_3, \ldots, x_{\mathrm{num\_steps} + 1}]$ of length ``num_steps``.
    The initial state $x_0$ and first difference $d_1$ are not part of the
    sample; they are supplied at construction and used to score the first
    post-initial transition. A random initial difference drawn from the
    stationary distribution can be handled by the calling temporal process as a
    separate sample site, matching the innovation parameterization.

    Parameters
    ----------
    autoreg
        AR(1) coefficient $\phi$ on first differences. For stationarity,
        $|\phi| < 1$; this is not enforced.
    scale
        Innovation standard deviation $\sigma$. Must be positive.
    initial_loc
        Deterministic initial state $x_0$. Not itself sampled. Defaults to
        ``0.0``.
    initial_diff
        Deterministic initial first difference $d_1 = x_1 - x_0$. Together with
        ``initial_loc`` it fixes $x_1 = x_0 + d_1$, used to score the first
        post-initial transition. Not itself sampled. Defaults to ``0.0``.
    num_steps
        Length of the post-initial path. Must be a positive integer.
    validate_args
        Forwarded to the base [numpyro.distributions.distribution.Distribution][].
    """

    arg_constraints = {
        "autoreg": constraints.real,
        "scale": constraints.positive,
        "initial_loc": constraints.real,
        "initial_diff": constraints.real,
    }
    support = constraints.real_vector
    reparametrized_params = ["autoreg", "scale", "initial_loc", "initial_diff"]
    pytree_aux_fields = ("num_steps",)

    def __init__(
        self,
        autoreg: ArrayLike,
        scale: ArrayLike,
        initial_loc: ArrayLike = 0.0,
        initial_diff: ArrayLike = 0.0,
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
        self.initial_diff = initial_diff
        self.num_steps = num_steps

        batch_shape = lax.broadcast_shapes(
            jnp.shape(autoreg),
            jnp.shape(scale),
            jnp.shape(initial_loc),
            jnp.shape(initial_diff),
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
        autoreg = jnp.asarray(self.autoreg)
        scale = jnp.asarray(self.scale)
        # lax.scan requires a shape-invariant carry; broadcast the two initial
        # states to per_step_shape so they match the per-step output shape.
        x0 = jnp.broadcast_to(jnp.asarray(self.initial_loc), per_step_shape)
        x1 = x0 + jnp.broadcast_to(jnp.asarray(self.initial_diff), per_step_shape)

        noise = random.normal(key, shape=(self.num_steps,) + per_step_shape)

        def step(
            carry: tuple[ArrayLike, ArrayLike], z_t: ArrayLike
        ) -> tuple[tuple[ArrayLike, ArrayLike], ArrayLike]:  # numpydoc ignore=GL08
            prev_2, prev_1 = carry
            new = prev_1 + autoreg * (prev_1 - prev_2) + scale * z_t
            return (prev_1, new), new

        _, xs = lax.scan(step, (x0, x1), noise)
        return jnp.moveaxis(xs, 0, -1)

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
        x0 = jnp.asarray(self.initial_loc)
        x1 = x0 + jnp.asarray(self.initial_diff)

        per_step_shape = jnp.shape(value)[:-1]
        x0 = jnp.broadcast_to(x0, per_step_shape)
        x1 = jnp.broadcast_to(x1, per_step_shape)

        # add length-1 trailing time dimension to fixed-in-time parameters
        x0_t = jnp.expand_dims(x0, -1)
        x1_t = jnp.expand_dims(x1, -1)
        scale_t = jnp.expand_dims(scale, -1)
        autoreg_t = jnp.expand_dims(autoreg, -1)

        v = jnp.concatenate([x0_t, x1_t, value], axis=-1)
        prev_delta = v[..., 1:-1] - v[..., :-2]
        means = v[..., 1:-1] + autoreg_t * prev_delta
        step_probs = Normal(means, scale_t).log_prob(v[..., 2:])
        return jnp.sum(step_probs, axis=-1)
