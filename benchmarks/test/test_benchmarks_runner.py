"""Tests for posterior summarization in ``benchmarks.core.runner``."""

import numpy as np
import pytest

from benchmarks.core.runner import summarize_posterior_parameters


class _FakeMcmc:
    """Minimal MCMC stub exposing ``get_samples`` for summarization tests."""

    def __init__(self, samples: dict):
        """Store the chain-grouped sample arrays to return."""
        self._samples = samples

    def get_samples(self, group_by_chain: bool = False) -> dict:
        """Return the stored samples.

        Returns
        -------
        dict
            Mapping from site name to a ``(chain, draw, ...)`` sample array.
        """
        return self._samples


def test_summarize_scalar_site_reports_mean_std_and_quantiles():
    """A scalar site reports posterior mean, std, and credible quantiles."""
    array = np.random.default_rng(0).normal(size=(2, 500))
    summaries = summarize_posterior_parameters(_FakeMcmc({"theta": array}))

    assert len(summaries) == 1
    s = summaries[0]
    assert s.site == "theta"
    assert s.mean == pytest.approx(float(np.mean(array)))
    assert s.std == pytest.approx(float(np.std(array)))
    assert s.q025 == pytest.approx(float(np.quantile(array, 0.025)))
    assert s.q25 == pytest.approx(float(np.quantile(array, 0.25)))
    assert s.q50 == pytest.approx(float(np.quantile(array, 0.5)))
    assert s.q75 == pytest.approx(float(np.quantile(array, 0.75)))
    assert s.q975 == pytest.approx(float(np.quantile(array, 0.975)))


def test_summarize_vector_site_reduces_per_element():
    """A vector site yields one summary per element over chains and draws."""
    array = np.random.default_rng(1).normal(size=(2, 300, 3))
    summaries = summarize_posterior_parameters(_FakeMcmc({"rt": array}))

    assert [s.index for s in summaries] == ["[0]", "[1]", "[2]"]
    for element, s in enumerate(summaries):
        per_element = array[:, :, element]
        assert s.mean == pytest.approx(float(np.mean(per_element)))
        assert s.std == pytest.approx(float(np.std(per_element)))
        assert s.q975 == pytest.approx(float(np.quantile(per_element, 0.975)))
