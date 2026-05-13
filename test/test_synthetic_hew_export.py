"""
Unit tests for exporting synthetic H+E data to a HEW model directory.
"""

import json
from pathlib import Path

import pytest

from pyrenew.datasets import write_synthetic_hew_model_dir


def test_write_synthetic_hew_model_dir_creates_expected_files(tmp_path: Path):
    """Exporter writes the model directory files expected by HEW fit helpers."""
    model_dir = write_synthetic_hew_model_dir(tmp_path / "synthetic_he")

    assert (model_dir / "priors.py").exists()
    assert (model_dir / "metadata.json").exists()
    assert (model_dir / "data" / "data_for_model_fit.json").exists()
    assert (model_dir / "data" / "model_params.json").exists()


def test_data_for_model_fit_has_production_he_schema(tmp_path: Path):
    """Exported data JSON has NSSP and NHSN training-data sections."""
    model_dir = write_synthetic_hew_model_dir(tmp_path / "synthetic_he")
    with open(model_dir / "data" / "data_for_model_fit.json") as f:
        data = json.load(f)

    assert data["loc_pop"] == [39512223]
    assert data["right_truncation_offset"] is None
    assert data["nssp_step_size"] == 1
    assert data["nhsn_step_size"] == 7

    nssp = data["nssp_training_data"]
    assert set(nssp) == {
        "date",
        "geo_value",
        "observed_ed_visits",
        "other_ed_visits",
        "data_type",
    }
    assert len(nssp["date"]) == 126
    assert len(nssp["observed_ed_visits"]) == 126

    nhsn = data["nhsn_training_data"]
    assert set(nhsn) == {
        "weekendingdate",
        "jurisdiction",
        "hospital_admissions",
        "data_type",
    }
    assert len(nhsn["weekendingdate"]) == 18
    assert len(nhsn["hospital_admissions"]) == 18


def test_model_params_has_production_he_schema(tmp_path: Path):
    """Exported model params JSON has fields required by PyrenewHEWParam."""
    model_dir = write_synthetic_hew_model_dir(tmp_path / "synthetic_he")
    with open(model_dir / "data" / "model_params.json") as f:
        params = json.load(f)

    assert set(params) == {
        "population_size",
        "pop_fraction",
        "generation_interval_pmf",
        "right_truncation_pmf",
        "inf_to_hosp_admit_lognormal_loc",
        "inf_to_hosp_admit_lognormal_scale",
        "inf_to_hosp_admit_pmf",
    }
    assert params["population_size"] == 39512223
    assert params["pop_fraction"] == [1.0]
    assert params["right_truncation_pmf"] == [1.0]
    assert len(params["generation_interval_pmf"]) == 7
    assert len(params["inf_to_hosp_admit_pmf"]) == 12
    assert params["inf_to_hosp_admit_lognormal_scale"] > 0


def test_model_params_can_use_hospital_delay(tmp_path: Path):
    """Exporter can write the example hospital delay PMF instead of ED delay."""
    model_dir = write_synthetic_hew_model_dir(
        tmp_path / "synthetic_he",
        delay_pmf_source="hospital",
    )
    with open(model_dir / "data" / "model_params.json") as f:
        params = json.load(f)

    assert len(params["inf_to_hosp_admit_pmf"]) > 12


def test_invalid_delay_pmf_source_raises(tmp_path: Path):
    """Exporter rejects unknown delay PMF source names."""
    with pytest.raises(ValueError, match="delay_pmf_source"):
        write_synthetic_hew_model_dir(
            tmp_path / "synthetic_he",
            delay_pmf_source="unknown",
        )


def test_overwrite_guard(tmp_path: Path):
    """Exporter refuses to replace an existing model directory by default."""
    model_dir = write_synthetic_hew_model_dir(tmp_path / "synthetic_he")

    with pytest.raises(FileExistsError):
        write_synthetic_hew_model_dir(model_dir)

    write_synthetic_hew_model_dir(model_dir, overwrite=True)
    assert (model_dir / "data" / "data_for_model_fit.json").exists()


def test_can_copy_external_priors_file(tmp_path: Path):
    """Exporter can copy a caller-provided priors file."""
    priors_path = tmp_path / "custom_priors.py"
    priors_path.write_text("# custom priors\n")

    model_dir = write_synthetic_hew_model_dir(
        tmp_path / "synthetic_he",
        priors_path=priors_path,
    )

    assert (model_dir / "priors.py").read_text() == "# custom priors\n"
