"""
Unit tests for the synthetic HEW model-directory export bridge.

These tests verify the on-disk layout and schema that the production
``pyrenew-multisignal`` / ``cfa-stf-routine-forecasting`` model builder
consumes, without importing either external package.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from pyrenew.datasets import (
    load_synthetic_daily_ed_visits,
    load_synthetic_weekly_hospital_admissions,
    write_synthetic_hew_model_dir,
)
from pyrenew.metaclass import RandomVariable

MODEL_PARAMS_KEYS = {
    "population_size",
    "pop_fraction",
    "generation_interval_pmf",
    "right_truncation_pmf",
    "inf_to_hosp_admit_lognormal_loc",
    "inf_to_hosp_admit_lognormal_scale",
    "inf_to_hosp_admit_pmf",
}

DATA_FOR_FIT_KEYS = {
    "loc_pop",
    "right_truncation_offset",
    "nssp_step_size",
    "nhsn_step_size",
    "nssp_training_data",
    "nssp_training_dates",
    "nhsn_training_data",
    "nhsn_training_dates",
}

REQUIRED_PRIOR_RVS = {
    "i0_first_obs_n_rv",
    "log_r_mu_intercept_rv",
    "eta_sd_rv",
    "autoreg_rt_rv",
    "ed_visit_wday_effect_rv",
    "ihr_rv",
}


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """Write a synthetic HEW model directory into a temporary path.

    Returns
    -------
    Path
        Path to the created model directory.
    """
    target = tmp_path / "synthetic_hew_model"
    return write_synthetic_hew_model_dir(target)


def _read_json(path: Path) -> dict:
    """Read and parse a JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    with open(path) as f:
        return json.load(f)


class TestWriteSyntheticHewModelDir:
    """Test write_synthetic_hew_model_dir output layout and schema."""

    def test_returns_model_dir_path(self, model_dir: Path) -> None:
        """Test that the function returns the created directory path."""
        assert model_dir.is_dir()

    def test_writes_expected_files(self, model_dir: Path) -> None:
        """Test that all files the production builder reads are written."""
        assert (model_dir / "priors.py").is_file()
        assert (model_dir / "data" / "data_for_model_fit.json").is_file()
        assert (model_dir / "data" / "model_params.json").is_file()
        assert (model_dir / "metadata.json").is_file()

    def test_model_params_schema(self, model_dir: Path) -> None:
        """Test that model_params.json carries the expected keys."""
        params = _read_json(model_dir / "data" / "model_params.json")
        assert set(params.keys()) == MODEL_PARAMS_KEYS

    def test_model_params_values(self, model_dir: Path) -> None:
        """Test that scalar model parameters are well formed."""
        params = _read_json(model_dir / "data" / "model_params.json")
        assert params["population_size"] > 0
        assert params["pop_fraction"] == [1.0]
        assert abs(sum(params["generation_interval_pmf"]) - 1.0) < 1e-6
        assert params["inf_to_hosp_admit_lognormal_scale"] > 0

    def test_data_for_fit_schema(self, model_dir: Path) -> None:
        """Test that data_for_model_fit.json carries the expected keys."""
        data = _read_json(model_dir / "data" / "data_for_model_fit.json")
        assert set(data.keys()) == DATA_FOR_FIT_KEYS

    def test_nssp_training_data_lengths_match(self, model_dir: Path) -> None:
        """Test that the daily ED columns align in length with the source data."""
        data = _read_json(model_dir / "data" / "data_for_model_fit.json")
        nssp = data["nssp_training_data"]
        n_days = len(load_synthetic_daily_ed_visits())
        for column in ("date", "geo_value", "observed_ed_visits", "other_ed_visits"):
            assert len(nssp[column]) == n_days

    def test_nhsn_training_data_lengths_match(self, model_dir: Path) -> None:
        """Test that the weekly hospital columns align with the source data."""
        data = _read_json(model_dir / "data" / "data_for_model_fit.json")
        nhsn = data["nhsn_training_data"]
        n_weeks = len(load_synthetic_weekly_hospital_admissions())
        for column in ("weekendingdate", "jurisdiction", "hospital_admissions"):
            assert len(nhsn[column]) == n_weeks

    def test_complete_reporting_defaults(self, model_dir: Path) -> None:
        """Test that defaults describe complete, non-truncated synthetic reports."""
        params = _read_json(model_dir / "data" / "model_params.json")
        data = _read_json(model_dir / "data" / "data_for_model_fit.json")
        assert params["right_truncation_pmf"] == [1.0]
        assert data["right_truncation_offset"] is None

    def test_priors_executes_and_defines_rvs(self, model_dir: Path) -> None:
        """Test that priors.py runs and defines the required random variables."""
        namespace = runpy.run_path(str(model_dir / "priors.py"))
        for name in REQUIRED_PRIOR_RVS:
            assert name in namespace, f"priors.py missing {name}"
            assert isinstance(namespace[name], RandomVariable)


class TestWriteSyntheticHewModelDirOptions:
    """Test optional behaviour of write_synthetic_hew_model_dir."""

    def test_raises_without_overwrite(self, model_dir: Path) -> None:
        """Test that writing into an existing directory raises by default."""
        with pytest.raises(FileExistsError):
            write_synthetic_hew_model_dir(model_dir)

    def test_overwrite_replaces_directory(self, model_dir: Path) -> None:
        """Test that overwrite=True replaces an existing directory."""
        stale = model_dir / "stale.txt"
        stale.write_text("stale")
        write_synthetic_hew_model_dir(model_dir, overwrite=True)
        assert not stale.exists()

    def test_invalid_delay_pmf_source_raises(self, tmp_path: Path) -> None:
        """Test that an unknown delay_pmf_source is rejected."""
        with pytest.raises(ValueError, match="delay_pmf_source"):
            write_synthetic_hew_model_dir(tmp_path / "bad", delay_pmf_source="invalid")

    def test_hospital_delay_source(self, tmp_path: Path) -> None:
        """Test that the hospital delay source selects a multi-day PMF."""
        target = write_synthetic_hew_model_dir(
            tmp_path / "hosp", delay_pmf_source="hospital"
        )
        params = _read_json(target / "data" / "model_params.json")
        assert len(params["inf_to_hosp_admit_pmf"]) > 1

    def test_right_truncation_offset_recorded(self, tmp_path: Path) -> None:
        """Test that a supplied right-truncation offset is written through."""
        target = write_synthetic_hew_model_dir(
            tmp_path / "rt", right_truncation_offset=3
        )
        data = _read_json(target / "data" / "data_for_model_fit.json")
        assert data["right_truncation_offset"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
