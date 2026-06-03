"""JAX and XLA environment configuration for benchmark drivers.

A driver must call :func:`configure_jax` before the first ``import jax`` in its
process so the float64 flag and host-device count take effect. This module
imports only the standard library, so importing it ahead of ``jax`` is safe.
"""

from __future__ import annotations

import os

AVAILABLE_CPUS: int = os.cpu_count() or 1
DEFAULT_DEVICE_COUNT: int = min(8, AVAILABLE_CPUS)
DEFAULT_NUM_CHAINS: int = min(4, AVAILABLE_CPUS)


def configure_jax() -> None:
    """Enable float64 and set the XLA host-device count if unset.

    Uses ``setdefault`` so an environment that already pins these flags is left
    untouched. Must run before the first ``import jax`` in the process.
    """
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault(
        "XLA_FLAGS",
        f"--xla_force_host_platform_device_count={DEFAULT_DEVICE_COUNT}",
    )
