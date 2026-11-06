"""Ejecución completa del notebook 09 en estado bloqueado y diagnóstico sintético."""
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
import pytest

from proyecto.inference import service
from proyecto.models import persistence
from proyecto.tracking import mlflow_utils
from tests.test_persistence import artifact

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / "notebooks/09_inferencia.ipynb").read_text(encoding="utf-8"))


def execute(monkeypatch, tmp_path, replacements=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RISKVISION_PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("RISKVISION_ENVIRONMENT_MODE", "local")
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    def forbidden(*a, **kw): pytest.fail("No training, export, datasets or remote services")
    monkeypatch.setattr(joblib, "dump", forbidden)
    monkeypatch.setattr(LGBMClassifier, "fit", forbidden)
    monkeypatch.setattr(persistence, "train_final_reference", forbidden)
    monkeypatch.setattr(service, "publish_predictions", forbidden)
    monkeypatch.setattr(mlflow_utils, "configure_mlflow", forbidden)
    ns = {}
    for i, cell in enumerate(NB["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            for old, new in (replacements or {}).items(): source = source.replace(old, new)
            exec(compile(source, f"09-cell-{i+1}", "exec"), ns)
    return ns


def test_default_notebook_no_reads_fits_or_writes(monkeypatch, tmp_path):
    def forbidden(*a, **kw): pytest.fail("No loading while controls are disabled")
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(service, "open_session", forbidden)
    monkeypatch.setattr(service, "predict_clients", forbidden)
    ns = execute(monkeypatch, tmp_path)
    assert not any(asdict(ns["CONTROLS"]).values())
    assert ns["SESSION"] is ns["REAL_RESULT"] is ns["BATCH_RESULT"] is None
    assert ns["REPORT_09"]["model_status"] == "provisional"
    assert ns["REPORT_09"]["real_metrics"] is None
    assert list(tmp_path.iterdir()) == []


def test_notebook_full_synthetic_diagnostic_using_08_bundle(artifact, monkeypatch, tmp_path):
    *_, receipt, bundle = artifact
    real_load = joblib.load
    def verified_bytes_only(file, *args, **kwargs):
        assert isinstance(file, BytesIO)  # no dataset reads, only verified serialized model bytes
        return real_load(file, *args, **kwargs)
    monkeypatch.setattr(joblib, "load", verified_bytes_only)
    ns = execute(monkeypatch, tmp_path, {
        "DIAGNOSTIC_MODE = False": "DIAGNOSTIC_MODE = True",
        "RUN_SYNTHETIC_EXAMPLES = False": "RUN_SYNTHETIC_EXAMPLES = True",
        "BUNDLE_DIRECTORY = None": f"BUNDLE_DIRECTORY = {receipt['directory']!r}",
        "EXPECTED_MANIFEST_SHA256 = None": f"EXPECTED_MANIFEST_SHA256 = {receipt['manifest_sha256']!r}",
        "TRUST_ARTIFACT_SOURCE = False": "TRUST_ARTIFACT_SOURCE = True",
        "SYNTHETIC_THRESHOLD = None": "SYNTHETIC_THRESHOLD = 0.5"})
    assert ns["SYNTHETIC_CONSISTENCY"]["threshold_applied"]
    assert ns["BATCH_RESULT"].index.tolist() == ["sintetico_A", "sintetico_B"]
    assert len(ns["SINGLE_RESULT"]) == len(ns["MISSING_RESULT"]) == 1
    assert ns["INVALID_INPUT_MESSAGE"]
    assert ns["REAL_RESULT"] is None
    assert ns["REPORT_09"]["diagnostic_only"] is True
    assert not ns["REPORT_09"]["real_publication_allowed"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle"]


def test_structure_and_no_training_test_or_registry_code():
    assert len(NB["cells"]) == 15
    for cell in NB["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "09", "exec")
            assert not any(token in source for token in (".fit(", ".optimize(", "test.joblib", "train.joblib", "joblib.load(", "register_model(", "@champion"))
            assert cell["outputs"] == [] and cell["execution_count"] is None
