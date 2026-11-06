"""Configuración del notebook 06 sin optimización, datos reales ni red."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import joblib
import mlflow
import pytest

from proyecto.tracking import mlflow_utils as mu

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "notebooks/06_optimizacion.ipynb").read_text(encoding="utf-8"))


def cell(number):
    return "".join(NOTEBOOK["cells"][number - 1]["source"])


def test_explicit_root_works_outside_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RISKVISION_PROJECT_ROOT", str(ROOT))
    ns = {}
    exec(cell(1), ns)
    assert ns["project_root"] == ROOT


@pytest.mark.parametrize("location", ["", "notebooks"])
def test_local_root_fallback(location, monkeypatch):
    monkeypatch.delenv("RISKVISION_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(ROOT / location)
    ns = {}
    exec(cell(1), ns)
    assert ns["project_root"] == ROOT


@pytest.mark.parametrize("location", ["relativa", "missing"])
def test_invalid_explicit_root_fails(tmp_path, monkeypatch, location):
    value = "relativa" if location == "relativa" else str(tmp_path / "missing")
    monkeypatch.setenv("RISKVISION_PROJECT_ROOT", value)
    with pytest.raises((ValueError, RuntimeError)):
        exec(cell(1), {})


def test_no_root_outside_project_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.delenv("RISKVISION_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="RISKVISION_PROJECT_ROOT"):
        exec(cell(1), {})


@pytest.mark.parametrize("filename", ["relative/train.joblib", "other.joblib", "missing/train.joblib"])
def test_invalid_training_location_fails_before_loading(tmp_path, monkeypatch, filename):
    value = filename if filename.startswith("relative") else str(tmp_path / filename)
    monkeypatch.setenv("RISKVISION_TRAIN_PATH", value)
    monkeypatch.setattr(joblib, "load", lambda *a: pytest.fail("No cargar datos inválidos"))
    with pytest.raises((ValueError, FileNotFoundError)):
        exec(cell(4), {"os": os, "Path": Path})


def test_explicit_training_location_uses_only_configured_file(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from proyecto.config import ID_COL, TARGET
    from proyecto.data.cleaning import PREDICTORS
    path = tmp_path / "train.joblib"
    path.write_bytes(b"fixture-sintetico")
    monkeypatch.setenv("RISKVISION_TRAIN_PATH", str(path))
    frame = pd.DataFrame({name: np.ones(24000) for name in PREDICTORS})
    frame[ID_COL] = np.arange(24000)
    frame[TARGET] = np.tile([0, 1], 12000)
    seen = []
    def fake_load(source):
        seen.append(source)
        return frame
    monkeypatch.setattr(joblib, "load", fake_load)
    ns = {"os": os, "Path": Path}
    exec(cell(4), ns)
    assert seen == [path]
    assert ns["X_all"].shape == (24000, 23)


def tracking_namespace(monkeypatch, workspace):
    monkeypatch.delenv("RISKVISION_TRACKING_MODE", raising=False)
    monkeypatch.setattr(mu, "is_databricks", lambda: workspace)
    calls = []
    monkeypatch.setattr(mu, "configure_mlflow", lambda **kw: calls.append(kw) or kw["backend"])
    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    ns = {"os": os, "project_root": ROOT}
    exec(cell(8), ns)
    return ns, calls


@pytest.mark.parametrize("workspace, expected", [(True, "databricks"), (False, "local")])
def test_tracking_preflight_selects_backend_without_creating_experiment(monkeypatch, workspace, expected):
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a: pytest.fail("No crear experimento en diagnóstico"))
    ns, calls = tracking_namespace(monkeypatch, workspace)
    assert calls == [{"backend": expected}]
    assert ns["experiment_id"] is None


def test_workspace_refuses_local_override(monkeypatch):
    monkeypatch.setenv("RISKVISION_TRACKING_MODE", "local")
    monkeypatch.setattr(mu, "is_databricks", lambda: True)
    with pytest.raises(ValueError, match="Databricks"):
        exec(cell(8), {"os": os, "project_root": ROOT})


def test_prepare_tracking_looks_up_returned_full_name(monkeypatch):
    ns, calls = tracking_namespace(monkeypatch, True)
    monkeypatch.setenv("EXPERIMENT_DIR", "/Shared/riskvision")
    monkeypatch.setattr(mu, "uses_databricks", lambda: True)
    created = []
    lookups = []
    monkeypatch.setattr(mlflow, "set_experiment", created.append)
    def lookup(name):
        lookups.append(name)
        return SimpleNamespace(experiment_id="123") if name in created else None
    monkeypatch.setattr(mlflow, "get_experiment_by_name", lookup)
    _, full_name, experiment_id = ns["prepare_tracking"]()
    assert full_name.startswith("/Shared/riskvision/riskvision-06-")
    assert lookups == [full_name, full_name]
    assert created == [full_name]
    assert experiment_id == "123"


def test_existing_experiment_is_not_reused(monkeypatch):
    ns, _ = tracking_namespace(monkeypatch, True)
    monkeypatch.setattr(ns["mlflow"], "get_experiment_by_name", lambda name: object())
    ns["resolve_experiment_name"] = lambda name: "/Shared/" + name
    ns["set_experiment"] = lambda name: pytest.fail("No sobrescribir")
    with pytest.raises(RuntimeError, match="ya existe"):
        ns["prepare_tracking"]()


def test_notebook_remains_disabled_and_code_compiles():
    for index, c in enumerate(NOTEBOOK["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"celda-{index + 1}", "exec")
    assert "RUN_OPTIMIZATION = False" in cell(6)
    assert "test.joblib" not in json.dumps(NOTEBOOK)
