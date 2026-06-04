"""Tests for the real-data feed parsing in ``benchmarks.core.real_data``.

These cover the one true external boundary: mapping live NSSP/NHSN feed frames
into a :class:`DatasetBundle`. The ``cfa.stf.data`` feed functions are faked
through ``sys.modules`` so no internal package or network access is required;
the tests pin the column schema the builders depend on, so an upstream schema
change is caught here.
"""

import sys
import types
from datetime import date

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from benchmarks.core.real_data import (
    RealDataSpec,
    _build_bundle,
    _build_ed_visits_signal,
    _build_hospital_signal,
)


def test_real_data_ed_signal_uses_current_nssp_schema(monkeypatch):
    """ED signal builder reads the current NSSP column schema."""
    calls = {}

    def get_nssp(**kwargs):
        """Return a minimal NSSP frame in the current schema.

        Returns
        -------
        polars.DataFrame
            Minimal NSSP rows for RSV and total ED visits.
        """
        calls.update(kwargs)
        return pl.DataFrame(
            {
                "reference_date": [
                    date(2025, 1, 1),
                    date(2025, 1, 1),
                    date(2025, 1, 2),
                    date(2025, 1, 2),
                ],
                "disease": ["RSV", "Total", "RSV", "Total"],
                "geo_value": ["US", "US", "US", "US"],
                "value": [10.0, 100.0, 12.0, 110.0],
            }
        )

    monkeypatch.setitem(
        sys.modules, "cfa.stf.data", types.SimpleNamespace(get_nssp=get_nssp)
    )

    signal = _build_ed_visits_signal(
        disease="RSV",
        loc_abbr="US",
        as_of=date(2025, 1, 10),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
        delay_pmf=jnp.array([1.0]),
    )

    assert calls["disease"] == ["RSV", "Total"]
    assert calls["lazy"] is False
    assert signal.start_date == date(2025, 1, 1)
    np.testing.assert_array_equal(np.asarray(signal.values), np.array([10.0, 12.0]))
    np.testing.assert_array_equal(
        np.asarray(signal.extras["other_ed_visits"]),
        np.array([90.0, 98.0]),
    )


def test_real_data_hospital_signal_uses_current_nhsn_schema(monkeypatch):
    """Hospital signal builder reads the current NHSN column schema."""
    calls = {}

    def get_nhsn_hrd(**kwargs):
        """Return a minimal NHSN HRD frame in the current schema.

        Returns
        -------
        polars.DataFrame
            Minimal NHSN hospital admission rows.
        """
        calls.update(kwargs)
        return pl.DataFrame(
            {
                "weekendingdate": [date(2025, 1, 4), date(2025, 1, 11)],
                "jurisdiction": ["US", "US"],
                "disease": ["RSV", "RSV"],
                "hospital_admissions": [40.0, 45.0],
            }
        )

    monkeypatch.setitem(
        sys.modules,
        "cfa.stf.data",
        types.SimpleNamespace(get_nhsn_hrd=get_nhsn_hrd),
    )

    signal = _build_hospital_signal(
        disease="RSV",
        loc_abbr="US",
        as_of=date(2025, 1, 15),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 14),
        delay_pmf=jnp.array([1.0]),
    )

    assert calls["lazy"] is False
    assert signal.start_date == date(2025, 1, 4)
    np.testing.assert_array_equal(np.asarray(signal.values), np.array([40.0, 45.0]))


def test_real_data_bundle_rejects_pre_nhsn_hospital_window():
    """Hospital bundles fail before feed calls when the window predates NHSN."""
    with pytest.raises(ValueError, match="as_of >= 2024-11-12"):
        _build_bundle(
            "real_he",
            RealDataSpec(
                disease="COVID-19",
                loc_abbr="US",
                as_of=date(2024, 11, 1),
                n_training_days=150,
                n_days_to_omit=2,
            ),
        )


def test_real_data_bundle_uses_static_references_and_live_he_feeds(monkeypatch):
    """Bundle setup uses local populations and live disease-specific PMFs."""
    calls = {"nssp": 0, "nhsn": 0, "gen_int": 0, "delay": 0}

    def get_nssp(**kwargs):  # numpydoc ignore=RT01
        """Return a minimal NSSP frame for bundle construction."""
        calls["nssp"] += 1
        return pl.DataFrame(
            {
                "reference_date": [
                    date(2025, 1, 1),
                    date(2025, 1, 1),
                    date(2025, 1, 2),
                    date(2025, 1, 2),
                ],
                "disease": ["RSV", "Total", "RSV", "Total"],
                "value": [10.0, 100.0, 12.0, 110.0],
            }
        )

    def get_nhsn_hrd(**kwargs):  # numpydoc ignore=RT01
        """Return a minimal NHSN frame for bundle construction."""
        calls["nhsn"] += 1
        return pl.DataFrame(
            {
                "weekendingdate": [date(2025, 1, 4)],
                "hospital_admissions": [40.0],
            }
        )

    def get_nnh_generation_interval_pmf(**kwargs):  # numpydoc ignore=RT01
        """Return a disease-specific generation interval test PMF."""
        calls["gen_int"] += 1
        assert kwargs["disease"] == "RSV"
        return [0.2, 0.8]

    def get_nnh_delay_pmf(**kwargs):  # numpydoc ignore=RT01
        """Return a disease-specific delay test PMF."""
        calls["delay"] += 1
        assert kwargs["disease"] == "RSV"
        return [0.1, 0.9]

    def fail_if_called(*args, **kwargs):
        """Fail if the old R location helper call reappears."""
        raise AssertionError("R forecasttools location helper should not be called")

    monkeypatch.setitem(
        sys.modules,
        "cfa.stf.data",
        types.SimpleNamespace(
            get_nssp=get_nssp,
            get_nhsn_hrd=get_nhsn_hrd,
            get_nnh_delay_pmf=get_nnh_delay_pmf,
            get_nnh_generation_interval_pmf=get_nnh_generation_interval_pmf,
            get_nnh_right_truncation_pmf=fail_if_called,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "cfa.stf.forecasttools",
        types.SimpleNamespace(get_us_loc_pop_tbl=fail_if_called),
    )

    bundle = _build_bundle(
        "real_he",
        RealDataSpec(
            disease="RSV",
            loc_abbr="CA",
            as_of=date(2025, 1, 10),
            n_training_days=2,
            n_days_to_omit=0,
        ),
    )

    assert calls == {"nssp": 1, "nhsn": 1, "gen_int": 1, "delay": 1}
    assert bundle.population_size == 39355309
    assert bundle.fixed_params == {}
    assert sorted(bundle.signals) == ["ed_visits", "hospital"]
    np.testing.assert_array_equal(np.asarray(bundle.gen_int_pmf), np.array([0.2, 0.8]))
    np.testing.assert_array_equal(
        np.asarray(bundle.signals["ed_visits"].extras["delay_pmf"]),
        np.array([0.1, 0.9]),
    )
