"""Informes y registro progresivo con datos sintéticos y almacenamiento temporal."""
from dataclasses import asdict
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import mlflow
import optuna
import pandas as pd
import pytest

from proyecto.models import tuning
from proyecto.tracking import reporting
from proyecto.tracking import mlflow_utils as mu
from tests.test_pipeline import _clients
from tests.test_tuning import _fake_scores


def test_code_revision_can_be_configured_when_git_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
    revision = reporting.code_revision(tmp_path, branch="revision", commit="abc123")
    assert revision == {"branch": "revision", "commit": "abc123", "source": "configured", "working_tree_status": None}
    assert reporting.code_revision(tmp_path)["commit"] is None


def test_real_local_tracking_statuses_and_reports_do_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mu.configure_mlflow(force_local=True, local_db=tmp_path / "tracking.db", env_file=tmp_path / "absent.env")
    mu.set_experiment("provisional-sintetico")
    experiment_id = mlflow.get_experiment_by_name("provisional-sintetico").experiment_id
    client = mlflow.MlflowClient()
    logger = tuning.MlflowTrialLogger(experiment_id, client=client)
    calls = []
    def scores(*args):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("fallo sintético")
        return _fake_scores(*args)
    monkeypatch.setattr(tuning, "score_folds", scores)
    config = tuning.TuningConfig(n_trials=2, n_splits=2, n_repeats=1)
    space = tuning.SearchSpace()
    study = tuning.run_search(*_clients(40), config=config, space=space, trial_logger=logger)
    for trial in study.trials:
        logger(study, trial)
    runs = client.search_runs([experiment_id])
    assert len(runs) == 2
    assert sorted(r.info.status for r in runs) == ["FAILED", "FINISHED"]
    report = reporting.build_provisional_report(
        revision={"branch": "sintetico", "commit": "abc", "source": "configured"},
        config=asdict(config), space=asdict(space), study=study, original_cv_auc=0.5,
        comparison={"original": 0.5, "optimized": 0.51},
        bootstrap={"delta_auc": 0.01, "low": -0.02, "high": 0.03},
        trial_run_ids=logger.run_ids, training_sha256="synthetic", reserve={"seed": 7, "share": .25},
        versions={"optuna": optuna.__version__}, experiment_name="provisional-sintetico",
        execution_started_at="2026-01-01T00:00:00+00:00")
    assert report["status"] == "pending_review" and not report["definitive"]
    assert report["metric"] == "roc_auc" and report["direction"] == "MAXIMIZE"
    assert report["trials"][0]["roc_auc_cv"] is None
    assert report["comparison_outcome"] == "no_clear_positive_difference"
    assert all(t["mlflow_run_id"] for t in report["trials"])
    first = reporting.log_provisional_report(report, experiment_id, client=client)
    second = reporting.log_provisional_report(report, experiment_id, client=client)
    assert first != second
    for run_id in [first, second]:
        destination = tmp_path / run_id
        destination.mkdir()
        artifact = client.download_artifacts(run_id, "reports/provisional.json", str(destination))
        assert json.loads(Path(artifact).read_text(encoding="utf-8")) == json.loads(json.dumps(report))
        assert client.get_run(run_id).data.tags["status"] == "pending_review"
    assert len(client.search_runs([experiment_id])) == 4


def test_report_rejects_nonfinite_values_before_creating_run():
    class ForbiddenClient:
        def create_run(self, *a, **kw):
            pytest.fail("No crear run para un informe inválido")
    with pytest.raises(ValueError):
        reporting.log_provisional_report({"report_id": "x", "value": float("nan")}, "1", client=ForbiddenClient())


def test_notebook_logs_trials_only_once_and_prepares_its_own_report(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/06_optimizacion.ipynb").read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    mu.configure_mlflow(force_local=True, local_db=tmp_path / "notebook.db", env_file=tmp_path / "absent.env")
    mu.set_experiment("notebook-sintetico")
    eid = mlflow.get_experiment_by_name("notebook-sintetico").experiment_id
    monkeypatch.setattr(tuning, "score_folds", _fake_scores)
    monkeypatch.setattr(reporting, "code_revision", lambda *a, **kw: {"branch": "sintetico", "commit": "abc"})
    ns = {"display": lambda *a: None, "pd": pd}
    exec("".join(notebook["cells"][5]["source"]), ns)
    X, y = _clients(80)
    ns.update(RUN_OPTIMIZATION=True, os=os, datetime=datetime, timezone=timezone,
              project_root=root, prepare_tracking=lambda: ("local", "notebook-sintetico", eid),
              CONFIG=tuning.TuningConfig(n_trials=2, n_splits=2, n_repeats=1), SPACE=tuning.SearchSpace(),
              X_search=X, y_search=y, RESERVE_SEED=7, RESERVE_SHARE=.25, TRAIN_SHA256="synthetic",
              score_folds=_fake_scores, log_run=mu.log_run, search_data=X, reserve_data=X.iloc[:20])
    exec("".join(notebook["cells"][13]["source"]), ns)
    exec("".join(notebook["cells"][17]["source"]), ns)
    runs = mlflow.MlflowClient().search_runs([eid])
    assert len(runs) == 4  # referencia + dos ensayos + informe; sin registro posterior duplicado
    assert len([r for r in runs if r.data.tags.get("role") == "optuna_trial"]) == 2
    assert ns["review_summary"]["code"]["branch"] == "sintetico"
    assert ns["review_summary"]["comparison_outcome"] == "pending"
