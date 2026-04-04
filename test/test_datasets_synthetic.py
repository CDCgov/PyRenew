"""
Unit tests for synthetic CA 120-day dataset loaders.
"""

import polars as pl
import pytest

from pyrenew.datasets import (
    load_synthetic_daily_ed_visits,
    load_synthetic_daily_hospital_admissions,
    load_synthetic_daily_infections,
    load_synthetic_true_parameters,
    load_synthetic_weekly_hospital_admissions,
)


class TestLoadSyntheticTrueParameters:
    """Test load_synthetic_true_parameters function."""

    def test_returns_dict(self):
        """Test that the loader returns a dictionary."""
        params = load_synthetic_true_parameters()
        assert isinstance(params, dict)

    def test_has_expected_keys(self):
        """Test that all expected top-level keys are present."""
        params = load_synthetic_true_parameters()
        expected_keys = {
            "description",
            "population",
            "start_date",
            "n_days",
            "n_init",
            "rng_seed",
            "generation_interval_pmf",
            "i0_per_capita",
            "rt_trajectory",
            "hospitalizations",
            "ed_visits",
        }
        assert set(params.keys()) == expected_keys

    def test_population_is_positive(self):
        """Test that population is a positive integer."""
        params = load_synthetic_true_parameters()
        assert params["population"] > 0

    def test_n_days_matches_data(self):
        """Test that n_days is 120."""
        params = load_synthetic_true_parameters()
        assert params["n_days"] == 120

    def test_hospitalization_params(self):
        """Test that hospitalization sub-dict has expected keys."""
        params = load_synthetic_true_parameters()
        hosp = params["hospitalizations"]
        assert "ihr" in hosp
        assert 0 < hosp["ihr"] < 1

    def test_ed_params(self):
        """Test that ED visit sub-dict has expected keys."""
        params = load_synthetic_true_parameters()
        ed = params["ed_visits"]
        assert "iedr" in ed
        assert "delay_pmf" in ed
        assert "day_of_week_effects" in ed
        assert 0 < ed["iedr"] < 1
        assert len(ed["day_of_week_effects"]) == 7


class TestLoadSyntheticDailyInfections:
    """Test load_synthetic_daily_infections function."""

    def test_returns_dataframe(self):
        """Test that the loader returns a Polars DataFrame."""
        df = load_synthetic_daily_infections()
        assert isinstance(df, pl.DataFrame)

    def test_has_expected_columns(self):
        """Test that expected columns are present."""
        df = load_synthetic_daily_infections()
        assert "date" in df.columns
        assert "true_infections" in df.columns
        assert "true_rt" in df.columns

    def test_has_120_rows(self):
        """Test that the dataset has 120 days."""
        df = load_synthetic_daily_infections()
        assert len(df) == 120

    def test_infections_are_positive(self):
        """Test that true infections are positive."""
        df = load_synthetic_daily_infections()
        assert (df["true_infections"] > 0).all()

    def test_rt_is_positive(self):
        """Test that true Rt values are positive."""
        df = load_synthetic_daily_infections()
        assert (df["true_rt"] > 0).all()


class TestLoadSyntheticDailyHospitalAdmissions:
    """Test load_synthetic_daily_hospital_admissions function."""

    def test_returns_dataframe(self):
        """Test that the loader returns a Polars DataFrame."""
        df = load_synthetic_daily_hospital_admissions()
        assert isinstance(df, pl.DataFrame)

    def test_has_expected_columns(self):
        """Test that expected columns are present."""
        df = load_synthetic_daily_hospital_admissions()
        assert "date" in df.columns
        assert "daily_hosp_admits" in df.columns
        assert "pop" in df.columns

    def test_has_120_rows(self):
        """Test that the dataset has 120 days."""
        df = load_synthetic_daily_hospital_admissions()
        assert len(df) == 120

    def test_admits_are_non_negative(self):
        """Test that hospital admissions are non-negative."""
        df = load_synthetic_daily_hospital_admissions()
        assert (df["daily_hosp_admits"] >= 0).all()


class TestLoadSyntheticDailyEdVisits:
    """Test load_synthetic_daily_ed_visits function."""

    def test_returns_dataframe(self):
        """Test that the loader returns a Polars DataFrame."""
        df = load_synthetic_daily_ed_visits()
        assert isinstance(df, pl.DataFrame)

    def test_has_expected_columns(self):
        """Test that expected columns are present."""
        df = load_synthetic_daily_ed_visits()
        assert "date" in df.columns
        assert "ed_visits" in df.columns

    def test_has_120_rows(self):
        """Test that the dataset has 120 days."""
        df = load_synthetic_daily_ed_visits()
        assert len(df) == 120

    def test_visits_are_non_negative(self):
        """Test that ED visits are non-negative."""
        df = load_synthetic_daily_ed_visits()
        assert (df["ed_visits"] >= 0).all()


class TestLoadSyntheticWeeklyHospitalAdmissions:
    """Test load_synthetic_weekly_hospital_admissions function."""

    def test_returns_dataframe(self):
        """Test that the loader returns a Polars DataFrame."""
        df = load_synthetic_weekly_hospital_admissions()
        assert isinstance(df, pl.DataFrame)

    def test_has_expected_columns(self):
        """Test that expected columns are present."""
        df = load_synthetic_weekly_hospital_admissions()
        assert "week_end" in df.columns
        assert "weekly_hosp_admits" in df.columns

    def test_admits_are_non_negative(self):
        """Test that weekly admissions are non-negative."""
        df = load_synthetic_weekly_hospital_admissions()
        assert (df["weekly_hosp_admits"] >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
