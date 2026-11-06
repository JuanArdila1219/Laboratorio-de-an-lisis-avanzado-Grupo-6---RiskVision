"""Informe provisional por ejecución, sin promover parámetros ni modelos."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4


def code_revision(root, *, branch=None, commit=None):
    """Consulta Git en lectura; permite metadatos explícitos en Databricks.

    Si no está disponible Git, no inventa la procedencia: devuelve None.
    Los valores explícitos se identifican como configuración del usuario.
    """
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(Path(root)), *args],
                                           stderr=subprocess.DEVNULL, text=True, timeout=5,
                                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"branch": branch or git("branch", "--show-current"),
            "commit": commit or git("rev-parse", "HEAD"),
            "source": "configured" if branch or commit else "git",
            "working_tree_status": git("status", "--porcelain", "--untracked-files=no")}


def build_provisional_report(*, revision, config, space, study, original_cv_auc,
                             comparison, bootstrap, trial_run_ids, training_sha256,
                             reserve, versions, experiment_name, execution_started_at):
    """Devuelve solo datos serializables y resultados de esta ejecución."""
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    report = {
        "report_id": uuid4().hex, "status": "pending_review", "definitive": False,
        "execution_started_at": execution_started_at,
        "report_created_at": datetime.now(timezone.utc).isoformat(), "code": dict(revision),
        "config": dict(config), "search_space": dict(space), "metric": "roc_auc",
        "direction": study.direction.name, "features": study.user_attrs.get("features"),
        "training_sha256": training_sha256, "reserve": dict(reserve), "versions": dict(versions),
        "experiment": experiment_name, "original_cv_auc": original_cv_auc,
        "best_trial": study.best_trial.number if completed else None,
        "best_cv_auc": study.best_value if completed else None,
        "best_params_candidate": dict(study.best_params) if completed else None,
        "internal_comparison": comparison, "paired_auc_bootstrap": bootstrap,
        "comparison_outcome": "pending" if bootstrap is None else (
            "conditional_positive_difference" if bootstrap["low"] > 0 else "no_clear_positive_difference"),
        "trials": [{"number": t.number, "state": t.state.name, "params": t.params,
                    "roc_auc_cv": t.value, "duration_seconds": None if t.duration is None else t.duration.total_seconds(),
                    "mlflow_run_id": trial_run_ids.get(t.number), "diagnostics": t.user_attrs}
                   for t in study.trials],
        "limitations": ["Reserva expuesta indirectamente en etapas anteriores; no evaluación independiente",
                        "Mejor AUC de búsqueda sujeto a optimismo de selección",
                        "Bootstrap condicionado a modelos ajustados; no incluye volver a entrenar o seleccionar",
                        "Penalización 5:1 secundaria e hipotética; no ahorro económico demostrado"],
    }
    # Rechaza NaN u objetos no serializables antes de crear un run.
    json.dumps(report, allow_nan=False)
    return report


def log_provisional_report(report, experiment_id, *, client=None):
    """Artefacto en un run nuevo: no sobrescribe informes de otros runs.

    Incluso al repetir esta función crea un run distinto. No registra modelos.
    """
    json.dumps(report, allow_nan=False)
    if client is None:
        from mlflow import MlflowClient
        client = MlflowClient()
    run = client.create_run(str(experiment_id), run_name=f"provisional-report-{uuid4().hex}",
                            tags={"role": "provisional_report", "status": "pending_review",
                                  "report_id": report["report_id"], "stage": "06"})
    try:
        client.log_dict(run.info.run_id, report, "reports/provisional.json")
        client.set_terminated(run.info.run_id, status="FINISHED")
    except BaseException:
        client.set_terminated(run.info.run_id, status="FAILED")
        raise
    return run.info.run_id
