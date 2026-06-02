"""Tests for the production HEW benchmark builder.

The ``pyrenew_multisignal`` and ``cfa-stf-routine-forecasting`` imports are
faked through ``sys.modules`` so these tests do not require either external
package, mirroring the real-data provider tests.
"""

import sys
import types
from pathlib import Path

import pytest

from benchmarks.core.hew_model import (
    HEW_RT_SITE_NAMES,
    _ensure_importable,
    build_hew_model,
    build_synthetic_hew_model,
)
from benchmarks.core.models import BuiltFit


class _FakeLatent:
    """Stand-in latent process exposing an initialization-point count."""

    n_initialization_points = 35


class _FakeModel:
    """Stand-in HEW model exposing the attributes the builder reads."""

    def __init__(self):
        """Attach a fake latent process."""
        self.latent_infection_process_rv = _FakeLatent()


@pytest.fixture
def fake_hew_imports(monkeypatch):
    """Install fake HEW external modules and capture their call arguments.

    Returns
    -------
    dict
        Captured keyword arguments from the data loader and model builder.
    """
    calls: dict[str, dict] = {}

    class FakeData:
        """Stand-in ``PyrenewHEWData`` with a recording ``from_json``."""

        @staticmethod
        def from_json(**kwargs):
            """Record loader kwargs and return a sentinel data object.

            Returns
            -------
            FakeData
                Sentinel data instance.
            """
            calls["data"] = kwargs
            return FakeData()

    def build_pyrenew_hew_model_from_dir(model_dir, **kwargs):
        """Record builder kwargs and return a fake model.

        Returns
        -------
        _FakeModel
            Stand-in HEW model.
        """
        calls["build"] = {"model_dir": model_dir, **kwargs}
        return _FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "pyrenew_multisignal.hew",
        types.SimpleNamespace(PyrenewHEWData=FakeData),
    )
    monkeypatch.setitem(
        sys.modules,
        "pipelines.utils.common_utils",
        types.SimpleNamespace(
            build_pyrenew_hew_model_from_dir=build_pyrenew_hew_model_from_dir
        ),
    )
    return calls


def test_rt_site_names_match_hew_model():
    """The HEW Rt site names match the model's deterministic registrations."""
    assert HEW_RT_SITE_NAMES == ("rt", "rtu_subpop")


def test_ensure_importable_prepends_paths(monkeypatch, tmp_path):
    """External repository paths are prepended to sys.path when absent."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    pms = tmp_path / "pyrenew-multisignal"
    cfa = tmp_path / "cfa-stf"
    _ensure_importable(pms, cfa)
    assert str(pms / "src") in sys.path
    assert str(cfa) in sys.path


def test_build_hew_model_returns_built_fit(fake_hew_imports, tmp_path):
    """The builder assembles a BuiltFit from the fake HEW model."""
    model_dir = tmp_path / "hew_model"
    built = build_hew_model(
        model_dir,
        pyrenew_multisignal_dir=tmp_path / "pms",
        cfa_stf_dir=tmp_path / "cfa",
    )

    assert isinstance(built, BuiltFit)
    assert isinstance(built.model, _FakeModel)
    assert built.dataset_name == "hew_model"
    assert built.n_initialization_points == 35


def test_build_hew_model_run_kwargs_carry_fit_flags(fake_hew_imports, tmp_path):
    """Run kwargs pass the data, signal flags, and NUTS step-size heuristic."""
    built = build_hew_model(
        tmp_path / "hew_model",
        fit_ed_visits=True,
        fit_hospital_admissions=True,
        fit_wastewater=False,
        pyrenew_multisignal_dir=tmp_path / "pms",
        cfa_stf_dir=tmp_path / "cfa",
    )

    assert built.run_kwargs["sample_ed_visits"] is True
    assert built.run_kwargs["sample_hospital_admissions"] is True
    assert built.run_kwargs["sample_wastewater"] is False
    assert built.run_kwargs["nuts_args"] == {"find_heuristic_step_size": True}
    assert "data" in built.run_kwargs


def test_build_hew_model_propagates_flags_to_externals(fake_hew_imports, tmp_path):
    """Signal flags reach both the data loader and the model builder."""
    build_hew_model(
        tmp_path / "hew_model",
        fit_ed_visits=True,
        fit_hospital_admissions=False,
        fit_wastewater=False,
        pyrenew_multisignal_dir=tmp_path / "pms",
        cfa_stf_dir=tmp_path / "cfa",
    )

    assert fake_hew_imports["data"]["fit_ed_visits"] is True
    assert fake_hew_imports["data"]["fit_hospital_admissions"] is False
    assert fake_hew_imports["build"]["fit_ed_visits"] is True
    assert fake_hew_imports["build"]["fit_hospital_admissions"] is False


def test_build_hew_model_dataset_name_override(fake_hew_imports, tmp_path):
    """An explicit dataset name overrides the model directory name."""
    built = build_hew_model(
        tmp_path / "hew_model",
        dataset_name="synthetic_he_weekly_hospital",
        pyrenew_multisignal_dir=tmp_path / "pms",
        cfa_stf_dir=tmp_path / "cfa",
    )
    assert built.dataset_name == "synthetic_he_weekly_hospital"


def test_build_synthetic_hew_model_writes_dir_then_builds(fake_hew_imports, tmp_path):
    """The synthetic helper writes a model directory and assembles from it."""
    model_dir = tmp_path / "synthetic_hew"
    built = build_synthetic_hew_model(
        model_dir,
        pyrenew_multisignal_dir=tmp_path / "pms",
        cfa_stf_dir=tmp_path / "cfa",
    )

    assert (model_dir / "data" / "data_for_model_fit.json").is_file()
    assert (model_dir / "priors.py").is_file()
    assert isinstance(built, BuiltFit)
    assert Path(fake_hew_imports["data"]["json_file_path"]).name == (
        "data_for_model_fit.json"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
