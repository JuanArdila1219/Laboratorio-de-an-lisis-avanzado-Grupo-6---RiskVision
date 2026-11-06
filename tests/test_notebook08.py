"""El notebook 08 debe ejecutarse sin lecturas de datos ni operaciones reales."""
from dataclasses import asdict
import json
from pathlib import Path

import joblib
import pytest

from proyecto.models import persistence
from proyecto.tracking import mlflow_utils, model_registry

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / "notebooks/08_registro_modelo.ipynb").read_text(encoding="utf-8"))


def test_disabled_notebook_in_new_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RISKVISION_PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("RISKVISION_ENVIRONMENT_MODE", "local")
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    def forbidden(*a, **kw): pytest.fail("No model fitting, data reads, persistence or publication allowed")
    for name in ("load_bundle", "save_bundle", "train_final_reference", "synthetic_persistence_check"):
        monkeypatch.setattr(persistence, name, forbidden)
    for name in ("publish_bundle_artifacts", "register_definitive_model", "promote_model_version"):
        monkeypatch.setattr(model_registry, name, forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(joblib, "dump", forbidden)
    monkeypatch.setattr(mlflow_utils, "configure_mlflow", forbidden)
    ns = {}
    for i, cell in enumerate(NB["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"08-cell-{i+1}", "exec"), ns)
    assert not any(asdict(ns["CONTROLS"]).values())
    assert ns["BUNDLE"] is ns["FITTED_MODEL"] is ns["REGISTRY_RECEIPT"] is None
    assert ns["REPORT_08"]["real_validation_metrics"] is None
    assert ns["REFERENCE_DECISION"]["status"] == "provisional"
    assert list(tmp_path.iterdir()) == []


def test_notebook_structure_no_outputs_and_no_dataset_reads():
    assert len(NB["cells"]) == 15
    for cell in NB["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            source = "".join(cell["source"])
            compile(source, "08", "exec")
            assert "test.joblib" not in source
            assert "joblib.load(" not in source
