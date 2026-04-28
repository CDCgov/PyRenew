"""
Unit tests for the datagen_he_CA_126 helper functions.
"""

import json
from datetime import date

import numpy as np
import polars as pl

import pyrenew.datasets.datagen_he_CA_126 as datagen_mod
from pyrenew.datasets.datagen_he_CA_126 import (
    apply_day_of_week_effects,
    build_true_rt,
    build_weekly_hosp_frame,
    generate,
    run_renewal,
    sample_negbinom,
)


class TestBuildTrueRt:
    """Tests for build_true_rt."""

    def test_length(self):
        """Test that the output has 126 days."""
        rt = build_true_rt()
        assert len(rt) == 126

    def test_starting_value(self):
        """Test that Rt starts near 1.2."""
        rt = build_true_rt()
        assert np.isclose(rt[0], 1.2, atol=0.01)

    def test_phase_endpoints(self):
        """Test that phase transitions occur at expected values."""
        rt = build_true_rt()
        # Phase 1 ends at index 62 (last day of 63-day decline), phase 2 starts at 63.
        assert rt[62] < 0.85
        assert rt[63] > 0.79


class TestRunRenewal:
    """Tests for run_renewal."""

    def test_output_length(self):
        """Test that output length equals n_init + len(rt)."""
        rt = np.ones(30)
        gen_int = np.array([0.5, 0.3, 0.2])
        result = run_renewal(rt, gen_int, i0_total=100.0, n_init=10)
        assert len(result) == 40

    def test_all_positive(self):
        """Test that all infections are positive."""
        rt = np.ones(30) * 1.1
        gen_int = np.array([0.6, 0.3, 0.1])
        result = run_renewal(rt, gen_int, i0_total=100.0, n_init=10)
        assert np.all(result > 0)

    def test_seed_period(self):
        """Test that the last seed value approximately equals i0_total."""
        rt = np.ones(20)
        gen_int = np.array([0.5, 0.3, 0.2])
        result = run_renewal(rt, gen_int, i0_total=500.0, n_init=5)
        assert np.isclose(result[4], 500.0, rtol=0.01)


class TestApplyDayOfWeekEffects:
    """Tests for apply_day_of_week_effects."""

    def test_uniform_effects(self):
        """Test that uniform effects leave values unchanged."""
        values = np.array([10.0, 20.0, 30.0])
        dow = np.ones(7)
        result = apply_day_of_week_effects(values, dow, first_day_dow=0)
        np.testing.assert_allclose(result, values)

    def test_known_pattern(self):
        """Test that known day-of-week effects are applied correctly."""
        values = np.ones(7) * 100.0
        dow = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
        result = apply_day_of_week_effects(values, dow, first_day_dow=0)
        np.testing.assert_allclose(result[5], 50.0)
        np.testing.assert_allclose(result[6], 50.0)

    def test_offset_start(self):
        """Test that first_day_dow correctly offsets the pattern."""
        values = np.ones(3) * 10.0
        dow = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        result = apply_day_of_week_effects(values, dow, first_day_dow=6)
        np.testing.assert_allclose(result[0], 10.0)
        np.testing.assert_allclose(result[1], 20.0)


class TestSampleNegbinom:
    """Tests for sample_negbinom."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        rng = np.random.default_rng(42)
        mu = np.array([100.0, 200.0, 300.0])
        result = sample_negbinom(mu, concentration=50.0, rng=rng)
        assert result.shape == mu.shape

    def test_mean_approximately_correct(self):
        """Test that sample mean is close to the specified mean for large n."""
        rng = np.random.default_rng(42)
        mu = np.full(10_000, 100.0)
        result = sample_negbinom(mu, concentration=500.0, rng=rng)
        assert np.isclose(np.mean(result), 100.0, rtol=0.05)

    def test_handles_near_zero_mu(self):
        """Test that near-zero mu values do not cause errors."""
        rng = np.random.default_rng(42)
        mu = np.array([0.0, 1e-15])
        result = sample_negbinom(mu, concentration=10.0, rng=rng)
        assert result.shape == mu.shape


class TestBuildWeeklyHospFrame:
    """Tests for build_weekly_hosp_frame."""

    def test_length_matches_input(self):
        """Frame length equals the number of weekly values supplied."""
        weekly = np.array([100.0, 200.0, 300.0])
        start = date(2023, 11, 6)
        result = build_weekly_hosp_frame(weekly, start)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_weekly_values_preserved(self):
        """Input weekly values appear in the weekly_hosp_admits column."""
        weekly = np.array([100.0, 200.0, 300.0])
        start = date(2023, 11, 6)
        result = build_weekly_hosp_frame(weekly, start)
        assert result["weekly_hosp_admits"].to_list() == [100.0, 200.0, 300.0]

    def test_first_week_end_is_saturday(self):
        """First week_end is the Saturday of the first complete MMWR epiweek."""
        # 2023-11-06 is a Monday; first MMWR Saturday after (Sun-start week) is 2023-11-18.
        weekly = np.array([1.0, 2.0])
        start = date(2023, 11, 6)
        result = build_weekly_hosp_frame(weekly, start)
        assert result["week_end"].to_list() == [date(2023, 11, 18), date(2023, 11, 25)]

    def test_first_week_end_sunday_start(self):
        """A Sunday start yields the immediately-following Saturday as first week_end."""
        # 2023-11-05 is a Sunday; first full MMWR week ends Saturday 2023-11-11.
        weekly = np.array([5.0])
        start = date(2023, 11, 5)
        result = build_weekly_hosp_frame(weekly, start)
        assert result["week_end"].to_list() == [date(2023, 11, 11)]


class TestGenerate:
    """Tests for the generate end-to-end function."""

    def test_generate_creates_expected_files(self, tmp_path, monkeypatch):
        """Test that generate() writes all expected output files."""
        monkeypatch.setattr(datagen_mod, "OUTPUT_DIR", tmp_path)
        generate()

        expected_files = [
            "true_parameters.json",
            "daily_infections.csv",
            "daily_ed_visits.csv",
            "daily_hospital_admissions.csv",
            "weekly_hospital_admissions.csv",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"Missing output file: {fname}"

    def test_generate_true_parameters_content(self, tmp_path, monkeypatch):
        """Test that true_parameters.json contains expected keys."""
        monkeypatch.setattr(datagen_mod, "OUTPUT_DIR", tmp_path)
        generate()

        with open(tmp_path / "true_parameters.json") as f:
            params = json.load(f)

        assert params["population"] == datagen_mod.POPULATION
        assert params["n_days"] == 126
        assert "hospitalizations" in params
        assert "ed_visits" in params

    def test_generate_daily_infections_shape(self, tmp_path, monkeypatch):
        """Test that daily_infections.csv has the correct number of rows."""
        monkeypatch.setattr(datagen_mod, "OUTPUT_DIR", tmp_path)
        generate()

        df = pl.read_csv(tmp_path / "daily_infections.csv")
        assert len(df) == 126
        assert "true_infections" in df.columns
        assert "true_rt" in df.columns

    def test_generate_hospital_admissions_shape(self, tmp_path, monkeypatch):
        """Test that daily_hospital_admissions.csv has the correct structure."""
        monkeypatch.setattr(datagen_mod, "OUTPUT_DIR", tmp_path)
        generate()

        df = pl.read_csv(tmp_path / "daily_hospital_admissions.csv")
        assert len(df) == 126
        assert "daily_hosp_admits" in df.columns

    def test_generate_ed_visits_shape(self, tmp_path, monkeypatch):
        """Test that daily_ed_visits.csv has the correct structure."""
        monkeypatch.setattr(datagen_mod, "OUTPUT_DIR", tmp_path)
        generate()

        df = pl.read_csv(tmp_path / "daily_ed_visits.csv")
        assert len(df) == 126
        assert "ed_visits" in df.columns
