"""Pytest configuration for the benchmark test suite.

Enables JAX float64 before any test imports ``jax``, so the benchmark tests run
in the same float64 mode as real benchmark runs. Without this, exact numeric
assertions on ``jnp`` arrays are import-order dependent (float32 vs float64).
``configure_jax`` only sets environment variables, so importing it here, ahead
of jax, is safe.
"""

from benchmarks.core.env import configure_jax

configure_jax()
