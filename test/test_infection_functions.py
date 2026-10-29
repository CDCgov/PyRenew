"""
Test functions from the latent.infection_functions
submodule
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import infection_functions as inf


def test_compute_infections_from_rt_with_feedback():
    """
    test that the implementation of infection
    feedback is as expected
    """

    # if feedback is zero, results should be
    # equivalent to compute_infections_from_rt
    # and Rt_adjusted should be Rt_raw

    gen_ints = [
        jnp.array([0.25, 0.5, 0.25]),
        jnp.array([1.0]),
        jnp.ones(35) / 35,
    ]

    inf_pmfs = [jnp.ones_like(x) for x in gen_ints]

    I0s = [
        jnp.array([0.235, 6.523, 100052.0]),
        jnp.array([5.0]),
        3.5235 * jnp.ones(35),
    ]

    Rts_raw = [
        jnp.array([1.25, 0.52, 23.0, 1.0]),
        jnp.ones(500),
        jnp.zeros(253),
    ]

    for I0, gen_int, inf_pmf in zip(I0s, gen_ints, inf_pmfs):
        for Rt_raw in Rts_raw:
            (
                infs_feedback,
                Rt_adj,
            ) = inf.compute_infections_from_rt_with_feedback(
                I0, Rt_raw, jnp.zeros_like(Rt_raw), gen_int, inf_pmf
            )

            assert_array_equal(
                inf.compute_infections_from_rt(I0, Rt_raw, gen_int),
                infs_feedback,
            )

            assert_array_equal(Rt_adj, Rt_raw)
    return None


@pytest.mark.parametrize(
    ["I0", "gen_int", "inf_pmf", "Rt_raw"],
    [
        [
            jnp.array([[5.0, 0.2]]),
            jnp.array([1.0]),
            jnp.array([1.0]),
            jnp.ones((5, 2)),
        ],
        [
            3.5235 * jnp.ones((35, 3)),
            jnp.ones(35) / 35,
            jnp.ones(35),
            jnp.zeros((253, 3)),
        ],
    ],
)
def test_compute_infections_from_rt_with_feedback_2d(I0, gen_int, inf_pmf, Rt_raw):
    """
    Test implementation of infection feedback
    when I0 and Rt are 2d arrays.
    """
    (
        infs_feedback,
        Rt_adj,
    ) = inf.compute_infections_from_rt_with_feedback(
        I0, Rt_raw, jnp.zeros_like(Rt_raw), gen_int, inf_pmf
    )

    assert_array_equal(
        inf.compute_infections_from_rt(I0, Rt_raw, gen_int),
        infs_feedback,
    )

    assert_array_equal(Rt_adj, Rt_raw)

    return None


def _roy_stan_loop(
    I0_counts: np.ndarray,
    Rt: np.ndarray,
    gen_int: np.ndarray,
    mobility_matrix: np.ndarray,
    pop: np.ndarray,
) -> np.ndarray:
    """
    Transcribe the time loop of the Roy et al. (2025) Stan model, in counts.

    Follows ``uk_spatial/simulated_data/simulated_fitting_region.stan`` in
    https://github.com/mlgh-sg/Spatial_transmission. The generation-interval
    sum runs over the full history, and the susceptibles of each region are
    its population minus every earlier day's infections.

    Parameters
    ----------
    I0_counts
        Infections on each initialization day, shape ``(n_init, n)``.
    Rt
        Reproduction number at each location after initialization,
        shape ``(n_days, n)``.
    gen_int
        Generation interval PMF; element 0 is one day back.
    mobility_matrix
        Column-stochastic mobility matrix, shape ``(n, n)``.
    pop
        Population of each region, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Infections after initialization, shape ``(n_days, n)``.
    """
    n_init, n = I0_counts.shape
    n_total = n_init + Rt.shape[0]
    infections = np.zeros((n_total, n))
    infections[:n_init] = I0_counts
    g = np.zeros(n_total + 1)
    g[1 : gen_int.size + 1] = gen_int
    eff_pop = mobility_matrix @ pop
    for t in range(n_init, n_total):
        convolution_inf = g[t - np.arange(t)] @ infections[:t]
        total_inf = mobility_matrix @ convolution_inf
        for m in range(n):
            sus = pop[m] - infections[:t, m].sum()
            infections[t, m] = np.dot(
                mobility_matrix[:, m], (sus / eff_pop) * Rt[t - n_init] * total_inf
            )
    return infections[n_init:]


def _eq1_loop(I0: np.ndarray, Rt: np.ndarray, gen_int: np.ndarray) -> np.ndarray:
    """
    Renew each region on its own with susceptible depletion, per resident.

    This is Eq 1 of Roy et al. (2025) for each region.

    Parameters
    ----------
    I0
        Infections per resident on each initialization day,
        shape ``(n_init, n)``.
    Rt
        Reproduction number of each region after initialization,
        shape ``(n_days, n)``.
    gen_int
        Generation interval PMF; element 0 is one day back.

    Returns
    -------
    np.ndarray
        Infections per resident after initialization, shape ``(n_days, n)``.
    """
    n_init, n = I0.shape
    n_total = n_init + Rt.shape[0]
    infections = np.zeros((n_total, n))
    infections[:n_init] = I0
    g = np.zeros(n_total + 1)
    g[1 : gen_int.size + 1] = gen_int
    for t in range(n_init, n_total):
        susceptible = 1.0 - infections[:t].sum(axis=0)
        infections[t] = (
            susceptible * Rt[t - n_init] * (g[t - np.arange(t)] @ infections[:t])
        )
    return infections[n_init:]


def test_mobility_matches_roy_stan_loop(
    gen_int_rv: DeterministicPMF,
    roy_mobility_matrix: jnp.ndarray,
    roy_subpop_fractions: jnp.ndarray,
    mobility_initial_infections: jnp.ndarray,
    mobility_rt_raw: jnp.ndarray,
) -> None:
    """Infections match the Roy et al. Stan time loop through heavy depletion."""
    pop = np.asarray(roy_subpop_fractions) * 45e6
    infections, _ = inf.compute_infections_from_rt_with_mobility(
        I0=mobility_initial_infections,
        Rt_raw=mobility_rt_raw,
        reversed_generation_interval_pmf=jnp.flip(gen_int_rv()),
        mobility_matrix=roy_mobility_matrix,
        subpop_fractions=roy_subpop_fractions,
    )
    expected = (
        _roy_stan_loop(
            np.asarray(mobility_initial_infections) * pop,
            np.asarray(mobility_rt_raw),
            np.asarray(gen_int_rv()),
            np.asarray(roy_mobility_matrix),
            pop,
        )
        / pop
    )

    assert_allclose(infections, expected, rtol=1e-10)
    depletion = mobility_initial_infections.sum(axis=0) + infections.sum(axis=0)
    assert jnp.all(depletion > 0.5)


def test_mobility_identity_without_depletion_matches_uncoupled(
    gen_int_rv: DeterministicPMF,
    roy_subpop_fractions: jnp.ndarray,
    mobility_initial_infections: jnp.ndarray,
    mobility_rt_raw: jnp.ndarray,
) -> None:
    """Identity matrix without depletion gives independent plain renewal."""
    reversed_gen_int = jnp.flip(gen_int_rv())
    infections, _ = inf.compute_infections_from_rt_with_mobility(
        I0=mobility_initial_infections,
        Rt_raw=mobility_rt_raw,
        reversed_generation_interval_pmf=reversed_gen_int,
        mobility_matrix=jnp.eye(3),
        subpop_fractions=roy_subpop_fractions,
        susceptible_depletion=False,
    )
    expected = inf.compute_infections_from_rt(
        I0=mobility_initial_infections[-reversed_gen_int.size :],
        Rt=mobility_rt_raw,
        reversed_generation_interval_pmf=reversed_gen_int,
    )

    assert_allclose(infections, expected, rtol=1e-12)


def test_mobility_identity_with_depletion_matches_eq1(
    gen_int_rv: DeterministicPMF,
    roy_subpop_fractions: jnp.ndarray,
    mobility_initial_infections: jnp.ndarray,
    mobility_rt_raw: jnp.ndarray,
) -> None:
    """Identity matrix with depletion gives Eq 1 for each region."""
    infections, _ = inf.compute_infections_from_rt_with_mobility(
        I0=mobility_initial_infections,
        Rt_raw=mobility_rt_raw,
        reversed_generation_interval_pmf=jnp.flip(gen_int_rv()),
        mobility_matrix=jnp.eye(3),
        subpop_fractions=roy_subpop_fractions,
    )
    expected = _eq1_loop(
        np.asarray(mobility_initial_infections),
        np.asarray(mobility_rt_raw),
        np.asarray(gen_int_rv()),
    )

    assert_allclose(infections, expected, rtol=1e-10)


def test_mobility_empty_location(
    empty_location_mobility: tuple[jnp.ndarray, jnp.ndarray],
    gen_int_rv: DeterministicPMF,
    mobility_initial_infections: jnp.ndarray,
    mobility_rt_raw: jnp.ndarray,
) -> None:
    """A location with no one present adds no infections, whatever its Rt, and its adjusted Rt is 0."""
    mobility_matrix, subpop_fractions = empty_location_mobility
    infections, rt_adjusted = inf.compute_infections_from_rt_with_mobility(
        I0=mobility_initial_infections,
        Rt_raw=mobility_rt_raw,
        reversed_generation_interval_pmf=jnp.flip(gen_int_rv()),
        mobility_matrix=mobility_matrix,
        subpop_fractions=subpop_fractions,
    )
    infections_other_rt, _ = inf.compute_infections_from_rt_with_mobility(
        I0=mobility_initial_infections,
        Rt_raw=mobility_rt_raw.at[:, 2].multiply(3.0),
        reversed_generation_interval_pmf=jnp.flip(gen_int_rv()),
        mobility_matrix=mobility_matrix,
        subpop_fractions=subpop_fractions,
    )

    assert jnp.all(jnp.isfinite(infections))
    assert_array_equal(infections_other_rt, infections)
    assert_array_equal(rt_adjusted[:, 2], 0.0)


def test_mobility_zero_population_region_matches_eq1(
    gen_int_rv: DeterministicPMF,
    mobility_initial_infections: jnp.ndarray,
    mobility_rt_raw: jnp.ndarray,
) -> None:
    """With the identity matrix, a region with no population leaves the others at Eq 1 and has no new infections."""
    infections, _ = inf.compute_infections_from_rt_with_mobility(
        I0=mobility_initial_infections,
        Rt_raw=mobility_rt_raw,
        reversed_generation_interval_pmf=jnp.flip(gen_int_rv()),
        mobility_matrix=jnp.eye(3),
        subpop_fractions=jnp.array([0.5, 0.5, 0.0]),
    )
    expected = _eq1_loop(
        np.asarray(mobility_initial_infections),
        np.asarray(mobility_rt_raw),
        np.asarray(gen_int_rv()),
    )

    assert_allclose(infections[:, :2], expected[:, :2], rtol=1e-10)
    assert_array_equal(infections[:, 2], 0.0)


def test_mobility_adjusted_rt_without_depletion(
    gen_int_rv: DeterministicPMF,
    roy_mobility_matrix: jnp.ndarray,
    roy_subpop_fractions: jnp.ndarray,
    mobility_initial_infections: jnp.ndarray,
    mobility_rt_raw: jnp.ndarray,
) -> None:
    """Without depletion, the adjusted Rt is the raw Rt."""
    _, rt_adjusted = inf.compute_infections_from_rt_with_mobility(
        I0=mobility_initial_infections,
        Rt_raw=mobility_rt_raw,
        reversed_generation_interval_pmf=jnp.flip(gen_int_rv()),
        mobility_matrix=roy_mobility_matrix,
        subpop_fractions=roy_subpop_fractions,
        susceptible_depletion=False,
    )

    assert_array_equal(rt_adjusted, mobility_rt_raw)


def test_mobility_conservation(
    gen_int_rv: DeterministicPMF,
    roy_mobility_matrix: jnp.ndarray,
    roy_subpop_fractions: jnp.ndarray,
    mobility_initial_infections: jnp.ndarray,
    mobility_rt_raw: jnp.ndarray,
) -> None:
    """Total new infections equal adjusted Rt times infectiousness present, summed over locations."""
    reversed_gen_int = np.asarray(jnp.flip(gen_int_rv()))
    infections, rt_adjusted = inf.compute_infections_from_rt_with_mobility(
        I0=mobility_initial_infections,
        Rt_raw=mobility_rt_raw,
        reversed_generation_interval_pmf=jnp.flip(gen_int_rv()),
        mobility_matrix=roy_mobility_matrix,
        subpop_fractions=roy_subpop_fractions,
    )
    n_init = mobility_initial_infections.shape[0]
    n_gen = reversed_gen_int.size
    history = np.vstack([mobility_initial_infections, infections])
    infectiousness = np.stack(
        [
            reversed_gen_int @ history[n_init + t - n_gen : n_init + t]
            for t in range(infections.shape[0])
        ]
    )
    p = np.asarray(roy_subpop_fractions)
    present = (infectiousness * p) @ np.asarray(roy_mobility_matrix).T

    assert_allclose(
        np.asarray(infections) @ p,
        np.sum(np.asarray(rt_adjusted) * present, axis=1),
        rtol=1e-12,
    )


def test_mobility_counts_all_initialization_days(
    gen_int_rv: DeterministicPMF,
    roy_mobility_matrix: jnp.ndarray,
    roy_subpop_fractions: jnp.ndarray,
) -> None:
    """Initialization days older than the generation interval count toward depletion."""
    reversed_gen_int = jnp.flip(gen_int_rv())
    recent = jnp.full((reversed_gen_int.size, 3), 1e-4)
    I0_low = jnp.vstack([jnp.full((13, 3), 1e-4), recent])
    I0_high = jnp.vstack([jnp.full((13, 3), 0.02), recent])

    first_day = []
    for I0 in (I0_low, I0_high):
        infections, _ = inf.compute_infections_from_rt_with_mobility(
            I0=I0,
            Rt_raw=jnp.full((5, 3), 1.5),
            reversed_generation_interval_pmf=reversed_gen_int,
            mobility_matrix=roy_mobility_matrix,
            subpop_fractions=roy_subpop_fractions,
        )
        first_day.append(infections[0])

    assert_allclose(
        first_day[1] / first_day[0],
        (1.0 - I0_high.sum(axis=0)) / (1.0 - I0_low.sum(axis=0)),
        rtol=1e-12,
    )


def test_mobility_extreme_rt_caps_incidence(
    gen_int_rv: DeterministicPMF,
    roy_mobility_matrix: jnp.ndarray,
    roy_subpop_fractions: jnp.ndarray,
    mobility_initial_infections: jnp.ndarray,
) -> None:
    """At extreme Rt, incidence stays within the remaining susceptible fraction."""
    infections, _ = inf.compute_infections_from_rt_with_mobility(
        I0=mobility_initial_infections,
        Rt_raw=jnp.full((60, 3), 30.0),
        reversed_generation_interval_pmf=jnp.flip(gen_int_rv()),
        mobility_matrix=roy_mobility_matrix,
        subpop_fractions=roy_subpop_fractions,
    )
    initial_cumulative = mobility_initial_infections.sum(axis=0)
    cumulative = initial_cumulative + jnp.cumsum(infections, axis=0)
    susceptible = 1.0 - jnp.vstack([initial_cumulative, cumulative[:-1]])

    assert jnp.all(infections >= 0)
    assert jnp.all(jnp.isfinite(infections))
    assert jnp.all(infections <= susceptible + 1e-12)
    assert jnp.all(cumulative <= 1.0 + 1e-12)
    assert_allclose(cumulative[-1], 1.0, atol=1e-12)


@pytest.mark.parametrize("susceptible_depletion", [True, False])
@pytest.mark.parametrize("rt", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("initial", [0.0, 0.2, 1.0])
def test_mobility_incidence_boundary(
    initial: float, rt: float, susceptible_depletion: bool
) -> None:
    """Jitted renewal caps only depleted incidence, including zero and full depletion."""
    compute = jax.jit(
        inf.compute_infections_from_rt_with_mobility,
        static_argnames="susceptible_depletion",
    )
    infections, rt_adjusted = compute(
        I0=jnp.array([[initial]]),
        Rt_raw=jnp.full((2, 1), rt),
        reversed_generation_interval_pmf=jnp.ones(1),
        mobility_matrix=jnp.ones((1, 1)),
        subpop_fractions=jnp.ones(1),
        susceptible_depletion=susceptible_depletion,
    )
    first = rt * initial
    if susceptible_depletion:
        first = min((1.0 - initial) * first, 1.0 - initial)
        susceptible = 1.0 - initial - first
        second = min(susceptible * rt * first, susceptible)
        expected_rt = [rt * (1.0 - initial), rt * susceptible]
    else:
        second = rt * first
        expected_rt = [rt, rt]

    assert_allclose(infections[:, 0], [first, second], atol=1e-12)
    assert_allclose(rt_adjusted[:, 0], expected_rt, atol=1e-12)


@pytest.mark.parametrize("susceptible_depletion", [True, False])
@pytest.mark.parametrize("invalid", [-0.1, jnp.nan, jnp.inf])
def test_mobility_rejects_invalid_initial_infections(
    invalid: float, susceptible_depletion: bool
) -> None:
    """Initialization must contain only finite, non-negative incident infections."""
    with pytest.raises(ValueError, match="finite, non-negative"):
        inf.compute_infections_from_rt_with_mobility(
            I0=jnp.array([[invalid], [0.1]]),
            Rt_raw=jnp.ones((1, 1)),
            reversed_generation_interval_pmf=jnp.ones(1),
            mobility_matrix=jnp.ones((1, 1)),
            subpop_fractions=jnp.ones(1),
            susceptible_depletion=susceptible_depletion,
        )


def test_mobility_initialization_cannot_exceed_population() -> None:
    """Every initialization day counts toward the per-region population bound."""
    kwargs = {
        "I0": jnp.array([[0.6, 0.1], [0.5, 0.1]]),
        "Rt_raw": jnp.ones((1, 2)),
        "reversed_generation_interval_pmf": jnp.ones(1),
        "mobility_matrix": jnp.eye(2),
        "subpop_fractions": jnp.full(2, 0.5),
    }
    with pytest.raises(ValueError, match="<= 1 in every region"):
        inf.compute_infections_from_rt_with_mobility(**kwargs)

    infections, _ = inf.compute_infections_from_rt_with_mobility(
        **kwargs, susceptible_depletion=False
    )
    assert_allclose(infections, [[0.5, 0.1]])


def test_mobility_gradient_finite(
    gen_int_rv: DeterministicPMF,
    roy_mobility_matrix: jnp.ndarray,
    roy_subpop_fractions: jnp.ndarray,
    mobility_initial_infections: jnp.ndarray,
) -> None:
    """Gradients with respect to Rt stay finite when the incidence ceiling binds."""

    def total_infections(rt: jnp.ndarray) -> jnp.ndarray:
        """
        Sum infections per resident over all days and regions.

        Parameters
        ----------
        rt
            Reproduction number at each location after initialization.

        Returns
        -------
        jnp.ndarray
            Scalar total.
        """
        infections, _ = inf.compute_infections_from_rt_with_mobility(
            I0=mobility_initial_infections,
            Rt_raw=rt,
            reversed_generation_interval_pmf=jnp.flip(gen_int_rv()),
            mobility_matrix=roy_mobility_matrix,
            subpop_fractions=roy_subpop_fractions,
        )
        return jnp.sum(infections)

    grad = jax.grad(total_infections)(jnp.full((60, 3), 30.0))

    assert jnp.all(jnp.isfinite(grad))
