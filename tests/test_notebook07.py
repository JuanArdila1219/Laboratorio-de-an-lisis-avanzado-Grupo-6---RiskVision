"""Integración del notebook 07 sin datos reales, ajustes, Optuna ni publicación."""
from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import mlflow
import pytest

from proyecto.config import ID_COL, TARGET
from proyecto.models import validation as v
from proyecto.tracking import mlflow_utils as mu, reporting
from tests.test_pipeline import _clients

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "notebooks/07_validacion_interpretacion.ipynb").read_text(encoding="utf-8"))


def run_disabled_notebook(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RISKVISION_PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("RISKVISION_ENVIRONMENT_MODE", "local")
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    frame, _ = _clients(96, seed=7)
    frame[ID_COL] = range(len(frame))
    frame[TARGET] = (frame["PAY_0"] >= 1).astype(int)
    frame.attrs.update(role="synthetic_training", source_sha256="synthetic")
    def fake_training(path, **kwargs):
        assert Path(path).name == "train.joblib"
        return frame
    def forbidden(*a, **kw):
        pytest.fail("No ajustar, calibrar, abrir prueba, exportar ni registrar con permisos desactivados")
    monkeypatch.setattr(v, "load_training", fake_training)
    for name in ("fit_training_candidate", "crossfit_probabilities", "evaluate_internal", "load_complementary_test",
                 "evaluate_complementary", "export_definitive_model", "interpretation"):
        monkeypatch.setattr(v, name, forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(joblib, "dump", forbidden)
    monkeypatch.setattr(mu, "configure_mlflow", forbidden)
    monkeypatch.setattr(reporting, "log_provisional_report", forbidden)
    monkeypatch.setattr(reporting, "code_revision", lambda *a, **kw: {"branch": "synthetic", "commit": "abc"})
    monkeypatch.setattr(mlflow, "start_run", forbidden)
    ns = {"display": lambda *a: None}
    for index, item in enumerate(NOTEBOOK["cells"]):
        if item["cell_type"] == "code":
            exec(compile("".join(item["source"]), f"cell-{index + 1}", "exec"), ns)
    return ns


def test_notebook_fresh_session_is_completely_blocked_without_temporary_cells(monkeypatch, tmp_path):
    ns = run_disabled_notebook(monkeypatch, tmp_path)
    assert asdict(ns["CONTROLS"]) == dict(fit_real=False, calibrate_real=False, evaluate_test=False, export_definitive=False)
    assert ns["INTERNAL"] is None and ns["COMPLEMENTARY"] is None
    assert ns["FROZEN_DECISION"] is None and ns["FITTED_FOR_EVALUATION"] == {}
    assert ns["REPORT_RUN_ID"] is None and "TEST_PATH" not in ns
    assert ns["REPORT_07"]["decision"]["status"] == "provisional"
    assert ns["REPORT_07"]["internal_evaluation"] is None
    json.dumps(ns["REPORT_07"], allow_nan=False)
    assert list(tmp_path.iterdir()) == []  # sin datos, modelos, experimentos o marcas de apertura


def test_enabling_only_test_flag_is_not_enough_to_open_it(monkeypatch, tmp_path):
    ns = run_disabled_notebook(monkeypatch, tmp_path)
    ns["CONTROLS"] = v.ExecutionControls(evaluate_test=True)
    with pytest.raises(PermissionError, match="Faltan decisiones"):
        exec("".join(NOTEBOOK["cells"][11]["source"]), ns)


def test_report_stage_is_explicit_without_changing_stage06_default():
    class Client:
        def __init__(self):
            self.calls = []
        def create_run(self, experiment_id, **kwargs):
            self.calls.append(deepcopy(kwargs))
            return SimpleNamespace(info=SimpleNamespace(run_id=str(len(self.calls))))
        def log_dict(self, *a, **kw):
            pass
        def set_terminated(self, *a, **kw):
            pass
    client = Client()
    reporting.log_provisional_report({"report_id": "synthetic"}, "1", client=client)
    reporting.log_provisional_report({"report_id": "synthetic"}, "1", client=client, stage="07")
    assert [call["tags"]["stage"] for call in client.calls] == ["06", "07"]


def test_notebook_has_valid_structure_and_all_controls_are_false():
    import nbformat
    nbformat.validate(nbformat.from_dict(NOTEBOOK))
    assert len(NOTEBOOK["cells"]) == 21
    source = "\n".join("".join(c["source"]) for c in NOTEBOOK["cells"] if c["cell_type"] == "code")
    for name in ("ALLOW_REAL_FIT", "ALLOW_REAL_CALIBRATION", "ALLOW_TEST_EVALUATION", "ALLOW_DEFINITIVE_EXPORT"):
        assert f"{name} = False" in source
    for c in NOTEBOOK["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "notebook07", "exec")
