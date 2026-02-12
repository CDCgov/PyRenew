"""
Unit tests for the pyrenew.convolve module
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import pyrenew.convolve as pc
import pyrenew.transformation as t

prng = np.random.RandomState(5)


@pytest.mark.parametrize(
    ["inits", "to_scan_a", "multipliers"],
    [
        [
            jnp.array([0.352, 5.2, -3]),
            jnp.array([0.5, 0.3, 0.2]),
            jnp.array(np.random.normal(0, 0.5, size=500)),
        ],
        [
            jnp.array(np.array([0.352, 5.2, -3] * 3).reshape(3, 3)),
            jnp.array([0.5, 0.3, 0.2]),
            jnp.array(np.random.normal(0, 0.5, size=(500, 3))),
        ],
    ],
)
def test_double_scanner_reduces_to_single(inits, to_scan_a, multipliers):
    """
    Test that new_double_scanner() yields a function
    that is equivalent to a single scanner if the first
    scan is chosen appropriately
    """

    def transform_a(x: any):
        """
        transformation associated with
        array to_scan_a

        Parameters
        ----------
        x
            input value

        Returns
        -------
        The result of 4 * x + 0.025, where x is the input
        value
        """
        return 4 * x + 0.025

    scanner_a = pc.new_convolve_scanner(to_scan_a, transform_a)

    double_scanner_a = pc.new_double_convolve_scanner(
        (jnp.array([523, 2, -0.5233]), to_scan_a),
        (jnp.ones_like, transform_a),
    )

    _, result_a = jax.lax.scan(f=scanner_a, init=inits, xs=multipliers)

    _, result_a_double = jax.lax.scan(
        f=double_scanner_a, init=inits, xs=(multipliers * 0.2352, multipliers)
    )

    assert_array_equal(result_a_double[1], jnp.ones_like(multipliers))
    assert_array_equal(result_a_double[0], result_a)


@pytest.mark.parametrize(
    ["arr", "history", "multipliers", "transform"],
    [
        [
            jnp.array([1.0, 2.0]),
            jnp.array([3.0, 4.0]),
            jnp.array([1, 2, 3]),
            t.IdentityTransform(),
        ],
        [
            jnp.ones(3),
            jnp.array(np.array([0.5, 0.3, 0.2] * 3)).reshape(3, 3),
            jnp.ones((3, 3)),
            t.ExpTransform(),
        ],
    ],
)
def test_convolve_scanner_using_scan(arr, history, multipliers, transform):
    """
    Tests the output of new convolve scanner function
    used with [`jax.lax.scan`][] against values calculated
    using a for loop
    """
    scanner = pc.new_convolve_scanner(arr, transform)

    _, result = jax.lax.scan(f=scanner, init=history, xs=multipliers)

    result_not_scanned = []
    for multiplier in multipliers:
        history, new_val = scanner(history, multiplier)
        result_not_scanned.append(new_val)

    assert jnp.array_equal(result, result_not_scanned)


@pytest.mark.parametrize(
    ["arr1", "arr2", "history", "m1", "m2", "transform"],
    [
        [
            jnp.array([1.0, 2.0]),
            jnp.array([2.0, 1.0]),
            jnp.array([0.1, 0.4]),
            jnp.array([1, 2, 3]),
            jnp.ones(3),
            (t.IdentityTransform(), t.IdentityTransform()),
        ],
        [
            jnp.array([1.0, 2.0, 0.3]),
            jnp.array([2.0, 1.0, 0.5]),
            jnp.array(np.array([0.5, 0.3, 0.2] * 3)).reshape(3, 3),
            jnp.ones((3, 3)),
            jnp.ones((3, 3)),
            (t.ExpTransform(), t.IdentityTransform()),
        ],
    ],
)
def test_double_convolve_scanner_using_scan(arr1, arr2, history, m1, m2, transform):
    """
    Tests the output of new convolve double scanner function
    used with [`jax.lax.scan`][] against values calculated
    using a for loop
    """
    arr1 = jnp.array([1.0, 2.0])
    arr2 = jnp.array([2.0, 1.0])
    transform = (t.IdentityTransform(), t.IdentityTransform())
    history = jnp.array([0.1, 0.4])
    m1, m2 = (jnp.array([1, 2, 3]), jnp.ones(3))

    scanner = pc.new_double_convolve_scanner((arr1, arr2), transform)

    _, result = jax.lax.scan(f=scanner, init=history, xs=(m1, m2))

    res1, res2 = [], []
    for m1, m2 in zip(m1, m2):
        history, new_val = scanner(history, (m1, m2))
        res1.append(new_val[0])
        res2.append(new_val[1])

    assert jnp.array_equal(result, (res1, res2))


@pytest.mark.parametrize(
    ["arr", "history", "multiplier", "transform"],
    [
        [
            jnp.array([1.0, 2.0]),
            jnp.array([3.0, 4.0]),
            jnp.array(2),
            t.IdentityTransform(),
        ],
        [
            jnp.ones(3),
            jnp.array(np.array([0.5, 0.3, 0.2] * 3)).reshape(3, 3),
            jnp.ones(3),
            t.ExpTransform(),
        ],
    ],
)
def test_convolve_scanner(arr, history, multiplier, transform):
    """
    Tests new convolve scanner function
    """
    scanner = pc.new_convolve_scanner(arr, transform)
    latest, new_val = scanner(history, multiplier)
    assert jnp.array_equal(new_val, transform(multiplier * jnp.dot(arr, history)))


@pytest.mark.parametrize(
    ["arr1", "arr2", "history", "m1", "m2", "transforms"],
    [
        [
            jnp.array([1.0, 2.0]),
            jnp.array([2.0, 1.0]),
            jnp.array([0.1, 0.4]),
            jnp.array(1),
            jnp.array(3),
            (t.IdentityTransform(), t.IdentityTransform()),
        ],
        [
            jnp.array([1.0, 2.0, 0.3]),
            jnp.array([2.0, 1.0, 0.5]),
            jnp.array(np.array([0.5, 0.3, 0.2] * 3)).reshape(3, 3),
            jnp.ones(3),
            0.1 * jnp.ones(3),
            (t.ExpTransform(), t.IdentityTransform()),
        ],
    ],
)
def test_double_convolve_scanner(arr1, arr2, history, m1, m2, transforms):
    """
    Tests new double convolve scanner function
    """
    double_scanner = pc.new_double_convolve_scanner((arr1, arr2), transforms)
    latest, (new_val, m_net) = double_scanner(history, (m1, m2))

    assert jnp.array_equal(m_net, transforms[0](m1 * jnp.dot(arr1, history)))
    assert jnp.array_equal(new_val, transforms[1](m2 * m_net * jnp.dot(arr2, history)))


@pytest.mark.parametrize(
    [
        "latent_incidence",
        "p_observed_given_incident",
        "unnormed_pmf",
    ],
    [
        [prng.random(size=100), 0.1, prng.random(size=25)],
        [prng.random(size=5), 0.1, prng.random(size=50)],
    ],
)
def test_compute_delay_ascertained_incidence(
    latent_incidence,
    p_observed_given_incident,
    unnormed_pmf,
):
    """
    Basic test that compute_delay_ascertained_incidence
    agrees with a manual reimplementation.
    """
    delay_incidence_to_observation_pmf = unnormed_pmf / np.sum(unnormed_pmf)
    # Expected results
    expected_output = jnp.convolve(
        p_observed_given_incident * latent_incidence,
        delay_incidence_to_observation_pmf,
        mode="valid",
    )
    expected_offset = len(delay_incidence_to_observation_pmf) - 1

    result, offset = pc.compute_delay_ascertained_incidence(
        latent_incidence,
        delay_incidence_to_observation_pmf,
        p_observed_given_incident,
    )
    assert_array_equal(result, expected_output)
    assert offset == expected_offset

    # Test pad=True
    result, offset = pc.compute_delay_ascertained_incidence(
        latent_incidence,
        delay_incidence_to_observation_pmf,
        p_observed_given_incident,
        pad=True,
    )
    assert_array_equal(
        result,
        jnp.pad(expected_output, (expected_offset, 0), constant_values=jnp.nan),
    )
    assert offset == 0


@pytest.mark.parametrize(
    [
        "latent_incidence",
        "p_observed_given_incident",
        "delay_incidence_to_observation_pmf",
        "error_type",
        "error_match",
    ],
    [
        [
            jnp.array([10, 20, 30, 40, 50, 60, 70, 80]),
            jnp.array([]),
            jnp.array([0.1, 0.2, 0.3, 0.5]),
            TypeError,
            "incompatible shapes",
        ],
        [
            jnp.array([]),
            0.25,
            jnp.array([1.0]),
            ValueError,
            "inputs cannot",
        ],
    ],
)
def test_compute_delay_ascertained_incidence_err(
    latent_incidence,
    p_observed_given_incident,
    delay_incidence_to_observation_pmf,
    error_type,
    error_match,
):
    """
    Test that compute_delay_ascertained_incidence
    errors as expected.
    """
    with pytest.raises(error_type, match=error_match):
        pc.compute_delay_ascertained_incidence(
            latent_incidence,
            delay_incidence_to_observation_pmf,
            p_observed_given_incident,
        )


@pytest.mark.parametrize(
    [
        "latent_incidence",
        "p_observed_given_incident",
        "delay_incidence_to_observation_pmf",
        "manual_expected_arr",
        "manual_expected_offset",
    ],
    [
        [
            jnp.array([10, 20, 30, 40, 50, 60, 70, 80]),
            0,
            jnp.array([0.1, 0.2, 0.3, 0.4]),
            jnp.zeros(5),
            3,
        ],
        [
            jnp.array([30]),
            1,
            jnp.array([0, 0, 0, 0, 1]),
            jnp.array([0, 0, 0, 0, 30]),
            4,
        ],
        [
            jnp.array([30, 40, 50, 60]),
            0,
            jnp.array([1]),
            jnp.array([0, 0, 0, 0]),
            0,
        ],
        [
            jnp.array([30, 40, 50, 60]),
            1,
            jnp.array([0, 1]),
            jnp.array([30, 40, 50]),
            1,
        ],
        [
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.0]),
            jnp.array([1.0]),
            jnp.array([1.0, 2.0, 3.0]),
            0,
        ],
        [
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.0, 0.1, 1.0]),
            jnp.array([1.0]),
            jnp.array([1.0, 0.2, 3.0]),
            0,
        ],
        [
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.0]),
            jnp.array([0.5, 0.5]),
            jnp.array([1.5, 2.5]),
            1,
        ],
        [
            jnp.array([0, 2.0, 4.0]),
            jnp.array([1.0]),
            jnp.array([0.25, 0.5, 0.25]),
            jnp.array([2]),
            2,
        ],
    ],
)
def test_compute_delay_ascertained_incidence_manual(
    latent_incidence,
    p_observed_given_incident,
    delay_incidence_to_observation_pmf,
    manual_expected_arr,
    manual_expected_offset,
):
    """
    Calculate some simple or reductive cases
    (e.g. p_obs = 0) manually.
    """
    result, offset = pc.compute_delay_ascertained_incidence(
        latent_incidence,
        delay_incidence_to_observation_pmf,
        p_observed_given_incident,
    )
    assert_array_equal(result, manual_expected_arr)
    assert offset == manual_expected_offset
    result_padded, offset_padded = pc.compute_delay_ascertained_incidence(
        latent_incidence,
        delay_incidence_to_observation_pmf,
        p_observed_given_incident,
        pad=True,
    )
    expected_padded = jnp.pad(
        1.0 * manual_expected_arr,  # ensure float
        pad_width=(manual_expected_offset, 0),
        mode="constant",
        constant_values=jnp.nan,
    )
    print(manual_expected_arr)
    print(expected_padded)
    assert offset_padded == 0
    assert_array_equal(result_padded, expected_padded)


@pytest.mark.parametrize(
    ["reporting_delay_pmf", "n_timepoints", "right_truncation_offset", "expected"],
    [
        [
            jnp.array([0.2, 0.3, 0.5]),
            5,
            0,
            jnp.array([1.0, 1.0, 1.0, 0.5, 0.2]),
        ],
        [
            jnp.array([0.2, 0.3, 0.5]),
            5,
            1,
            jnp.array([1.0, 1.0, 1.0, 1.0, 0.5]),
        ],
        [
            jnp.array([0.2, 0.3, 0.5]),
            5,
            10,
            jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        ],
        [
            jnp.array([0.2, 0.3, 0.5]),
            2,
            0,
            jnp.array([0.5, 0.2]),
        ],
        [
            jnp.array([1.0]),
            3,
            0,
            jnp.array([1.0, 1.0, 1.0]),
        ],
    ],
)
def test_compute_prop_already_reported(
    reporting_delay_pmf,
    n_timepoints,
    right_truncation_offset,
    expected,
):
    """
    Test compute_prop_already_reported against hand-calculated values.

    PMF [0.2, 0.3, 0.5] has CDF [0.2, 0.5, 1.0].

    offset=0: tail = flip(CDF[0:]) = [1.0, 0.5, 0.2]
    offset=1: tail = flip(CDF[1:]) = [1.0, 0.5]
    offset=10: tail = flip(CDF[10:]) = [] (empty, all padded to 1.0)
    """
    result = pc.compute_prop_already_reported(
        reporting_delay_pmf, n_timepoints, right_truncation_offset
    )
    assert result.shape == (n_timepoints,)
    assert_array_equal(result, expected)
