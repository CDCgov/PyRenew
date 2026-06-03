"""Builder for the production HEW model as a benchmark candidate.

Wraps the ``pyrenew-multisignal`` HEW model in a :class:`BuiltFit` so the
shared runner in :mod:`benchmarks.core.runner` fits it exactly like a PyRenew
:class:`MultiSignalModel`: ``model.run`` with the benchmark's diagnostic
``extra_fields``, then metrics read from ``model.mcmc``.

The HEW model is fit in process rather than through the pipeline's
``fit_and_save_model``. That entry point pickles to disk and requests
``extra_fields`` that omit ``diverging`` and ``energy``, which the benchmark
needs for divergence and E-BFMI metrics. Building and running the model
directly lets the runner request the diagnostic fields it reports.

Imports of ``pyrenew_multisignal`` and the ``cfa-stf-routine-forecasting``
pipeline utilities are deferred to call time, mirroring
:mod:`benchmarks.core.real_data`, so importing this module does not require
either package. Install them separately to use this builder.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

from benchmarks.core.models import BuiltFit
from benchmarks.core.signals import DatasetBundle
from pyrenew.datasets import write_hew_model_dir, write_synthetic_hew_model_dir

HEW_RT_SITE_NAMES: tuple[str, ...] = ("rt", "rtu_subpop")
"""Posterior site names carrying the HEW Rt trajectory, in priority order."""

DEFAULT_PYRENEW_MULTISIGNAL_DIR: Path = Path(
    "~/github/CDC/pyrenew-multisignal"
).expanduser()
DEFAULT_CFA_STF_DIR: Path = Path(
    "~/github/CDC/cfa-stf-routine-forecasting"
).expanduser()


def _ensure_importable(
    pyrenew_multisignal_dir: str | Path, cfa_stf_dir: str | Path
) -> None:
    """Prepend the external repository paths to ``sys.path`` if absent."""
    for path in (Path(pyrenew_multisignal_dir) / "src", Path(cfa_stf_dir)):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def build_hew_model(
    model_dir: str | Path,
    *,
    fit_ed_visits: bool = True,
    fit_hospital_admissions: bool = True,
    fit_wastewater: bool = False,
    dataset_name: str | None = None,
    pyrenew_multisignal_dir: str | Path = DEFAULT_PYRENEW_MULTISIGNAL_DIR,
    cfa_stf_dir: str | Path = DEFAULT_CFA_STF_DIR,
) -> BuiltFit:
    """Assemble the production HEW model from a model directory.

    Parameters
    ----------
    model_dir
        Directory in the production HEW layout, containing ``priors.py`` and
        ``data/`` as written by
        :func:`pyrenew.datasets.write_synthetic_hew_model_dir`.
    fit_ed_visits
        Whether to fit the ED-visits signal.
    fit_hospital_admissions
        Whether to fit the hospital-admissions signal.
    fit_wastewater
        Whether to fit the wastewater signal.
    dataset_name
        Identifier reported in benchmark output. Defaults to the model
        directory name. Set it to match a paired PyRenew candidate's dataset
        so the two share a comparison group.
    pyrenew_multisignal_dir
        Path to the ``pyrenew-multisignal`` checkout.
    cfa_stf_dir
        Path to the ``cfa-stf-routine-forecasting`` checkout.

    Returns
    -------
    BuiltFit
        Model and run kwargs ready for :func:`benchmarks.core.runner.fit_and_measure`.
    """
    _ensure_importable(pyrenew_multisignal_dir, cfa_stf_dir)
    from pipelines.utils.common_utils import build_pyrenew_hew_model_from_dir
    from pyrenew_multisignal.hew import PyrenewHEWData

    model_dir = Path(model_dir)
    data = PyrenewHEWData.from_json(
        json_file_path=model_dir / "data" / "data_for_model_fit.json",
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
    )
    model = build_pyrenew_hew_model_from_dir(
        model_dir,
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
    )
    run_kwargs = {
        "data": data,
        "sample_ed_visits": fit_ed_visits,
        "sample_hospital_admissions": fit_hospital_admissions,
        "sample_wastewater": fit_wastewater,
        "nuts_args": {"find_heuristic_step_size": True},
    }
    return BuiltFit(
        model=model,
        run_kwargs=run_kwargs,
        dataset_name=dataset_name or model_dir.name,
        n_initialization_points=(
            model.latent_infection_process_rv.n_initialization_points
        ),
    )


def build_synthetic_hew_model(
    model_dir: str | Path,
    *,
    overwrite: bool = True,
    dataset_name: str | None = None,
    fit_ed_visits: bool = True,
    fit_hospital_admissions: bool = True,
    fit_wastewater: bool = False,
    pyrenew_multisignal_dir: str | Path = DEFAULT_PYRENEW_MULTISIGNAL_DIR,
    cfa_stf_dir: str | Path = DEFAULT_CFA_STF_DIR,
) -> BuiltFit:
    """Write a synthetic HEW model directory and assemble the model from it.

    Returns
    -------
    BuiltFit
        Model and run kwargs ready for :func:`benchmarks.core.runner.fit_and_measure`.
    """
    write_synthetic_hew_model_dir(model_dir, overwrite=overwrite)
    return build_hew_model(
        model_dir,
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
        dataset_name=dataset_name,
        pyrenew_multisignal_dir=pyrenew_multisignal_dir,
        cfa_stf_dir=cfa_stf_dir,
    )


def write_hew_model_dir_from_bundle(
    bundle: DatasetBundle,
    model_dir: str | Path,
    *,
    location: str,
    disease: str,
    overwrite: bool = True,
    right_truncation_offset: int | None = None,
    right_truncation_pmf: list[float] | None = None,
) -> Path:
    """Serialize a :class:`DatasetBundle` into the production HEW model directory.

    Lets the HEW arm fit the same H+E feeds as a PyRenew candidate built from
    the same bundle. The bundle must carry a daily ``ed_visits`` signal and a
    weekly ``hospital`` signal; the ED signal must supply ``other_ed_visits``
    in its ``extras`` and the hospital signal a ``delay_pmf`` for the
    infection-to-admission delay.

    Parameters
    ----------
    bundle
        H+E dataset bundle, as returned by
        :func:`benchmarks.core.data_source.load_he_bundle`.
    model_dir
        Directory to write the production HEW layout into.
    location
        Jurisdiction code written into the production-shaped data.
    disease
        Disease label recorded in ``metadata.json``.
    overwrite
        Whether to replace an existing ``model_dir``.
    right_truncation_offset
        Offset for the production ED right-truncation adjustment.
    right_truncation_pmf
        Reporting-delay PMF. Defaults to ``[1.0]`` for complete reports.

    Returns
    -------
    pathlib.Path
        Path to the created model directory.
    """
    ed = bundle.signals["ed_visits"]
    hosp = bundle.signals["hospital"]
    if "other_ed_visits" not in ed.extras:
        raise KeyError(
            "ed_visits signal must carry 'other_ed_visits' in extras to write a "
            "HEW model directory."
        )
    if "delay_pmf" not in hosp.extras:
        raise KeyError(
            "hospital signal must carry 'delay_pmf' in extras to write a HEW "
            "model directory."
        )
    ed_dates = [
        (ed.start_date + dt.timedelta(days=i)).isoformat()
        for i in range(len(ed.values))
    ]
    hosp_dates = [
        (hosp.start_date + dt.timedelta(weeks=i)).isoformat()
        for i in range(len(hosp.values))
    ]
    return write_hew_model_dir(
        model_dir,
        ed_dates=ed_dates,
        observed_ed_visits=[float(x) for x in ed.values],
        other_ed_visits=[float(x) for x in ed.extras["other_ed_visits"]],
        hosp_dates=hosp_dates,
        hospital_admissions=[float(x) for x in hosp.values],
        population=int(bundle.population_size),
        generation_interval_pmf=[float(x) for x in bundle.gen_int_pmf],
        inf_to_hosp_admit_pmf=[float(x) for x in hosp.extras["delay_pmf"]],
        location=location,
        disease=disease,
        overwrite=overwrite,
        right_truncation_offset=right_truncation_offset,
        right_truncation_pmf=right_truncation_pmf,
        source="benchmarks.core.hew_model.write_hew_model_dir_from_bundle",
    )
