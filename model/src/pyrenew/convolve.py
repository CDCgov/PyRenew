"""
convolve

Factory functions for
calculating convolutions of timeseries
with discrete distributions
of times-to-event using
jax.lax.scan. Factories generate functions
that can be passed to scan with an
appropriate array to scan.
"""
import jax.numpy as jnp


def new_convolve_scanner(discrete_dist_flipped):
    def _new_scanner(history_subset, multiplier):
        new_val = multiplier * jnp.dot(discrete_dist_flipped, history_subset)
        latest = jnp.hstack([history_subset[1:], new_val])
        return latest, new_val

    return _new_scanner


def new_double_scanner(dists, transforms):
    d1, d2 = dists
    t1, t2 = transforms

    def _new_scanner(history_subset, multipliers):
        m1, m2 = multipliers
        m_net1 = t1(m1 * jnp.dot(d1, history_subset))
        new_val = t2(m2 * m_net1 * jnp.dot(d2, history_subset))
        latest = jnp.hstack([history_subset[1:], new_val])
        return (latest, (new_val, m_net1))

    return _new_scanner
