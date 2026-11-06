"""Decisión JSON entre 07, 08 y 09; autorizaciones humanas separadas del cálculo."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from proyecto.data.cleaning import PREDICTORS
from proyecto.features.build import PACKAGE_E
from proyecto.models.candidates import LGBM_PARAMS

MODEL = "LightGBM original"
ESTEFANIA_COMMIT = "e4067168e4f75dfe3c4f9f5e83cf20df797775c5"
ADOPTED_PARAMS = dict(n_estimators=300, learning_rate=.03, num_leaves=15, min_child_samples=50,
    subsample=.8, subsample_freq=1, colsample_bytree=.8, random_state=42, verbose=-1, n_jobs=4)


def adopted_experiment():
    """Resultados REPORTADOS, no una reproducción ni aprobación de rendimiento."""
    if ADOPTED_PARAMS != LGBM_PARAMS:
        raise ValueError("Los parámetros actuales difieren de la configuración conservada por Estefanía")
    return {"author": "Estefanía", "branch": "origin/Estefania", "commit": ESTEFANIA_COMMIT,
        "sources": ["reports/06_best_params.json", "reports/06_optimizacion.md", "notebooks/06_optimizacion.ipynb"],
        "team_decision": "Adoptar la configuración original como candidata; pendiente de validación propia",
        "model": MODEL, "features": list(PACKAGE_E), "params": deepcopy(LGBM_PARAMS),
        "own_real_optimization_executed": False, "evidence_status": "reported_by_estefania_not_reexecuted",
        "experiment": "/Users/estefaniacg1995@gmail.com/riskvision-optimizacion",
        "reported_metrics": {"n_trials": 12, "best_cv_auc": .7869,
            "reserve_auc_rounded": {"original": .7865, "optimized": .7865},
            "reserve_penalty_rounded": {"original": .549, "optimized": .543},
            "penalty_difference_ci95": [-.0185, .007166666666666667]},
        "decision_rule": "Conservar original si el IC de penalización optimizado-original no queda íntegramente bajo cero",
        "penalty_assumption": {"false_negative": 5, "false_positive": 1, "threshold_used": 1 / 6},
        "limitations": ["Métricas de otro experimento; no son validación propia de 07",
            "Reserva y prueba original expuestas previamente; no evaluación independiente",
            "IC de penalización, no de AUC; cruzar cero no demuestra equivalencia",
            "Costos 5:1 hipotéticos, no ahorro económico demostrado",
            "Sin huella de entrenamiento, versiones ni run_id completos en los informes publicados",
            "Espacio de búsqueda y configuración de ejecución distintos del módulo tuning.py propio",
            "No hay evidencia ejecutada propia de Optuna ni aprobación definitiva"]}


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _utc(value):
    if not isinstance(value, str):
        raise ValueError("La aprobación requiere una fecha UTC explícita")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("La fecha de aprobación debe incluir UTC")


def validate_handoff(decision):
    """Contrato estricto para el ajuste final; no cambia estado ni otorga permisos."""
    contract = decision.get("handoff", {})
    if (decision.get("model") != MODEL or decision.get("params") != LGBM_PARAMS
            or decision.get("features") != list(PACKAGE_E)
            or decision.get("own_real_optimization_executed") is not False
            or decision.get("antecedent") != adopted_experiment()
            or decision.get("frozen_for_evaluation") is not True
            or contract.get("format_version") != 1 or contract.get("final_fit_owner") != "08"
            or contract.get("input_columns") != PREDICTORS
            or contract.get("expected_training_shape") != [24000, 25]):
        raise ValueError("Contrato 07→08 incompatible con la referencia adoptada")
    if decision.get("status") not in {"provisional", "approved_definitive"}:
        raise ValueError("Estado de decisión inválido")
    if not re.fullmatch(r"[a-f0-9]{64}", decision.get("training_sha256", "")):
        raise ValueError("Falta SHA-256 del entrenamiento")
    if not all(decision.get("code", {}).get(key) for key in ("branch", "commit")) or not decision.get("versions"):
        raise ValueError("Faltan revisión y versiones de validación")
    choice = decision.get("settings", {}).get(MODEL, {})
    threshold = choice.get("threshold")
    if (choice.get("calibration") not in {"none", "sigmoid"}
            or type(threshold) not in (int, float) or not 0 < threshold < 1):
        raise ValueError("Calibración o umbral incompatibles con inferencia")
    report = contract.get("validation_report", {})
    selected = MODEL + (" + sigmoid" if choice.get("calibration") == "sigmoid" else "")
    if (report.get("stage") != "07" or not report.get("report_id")
            or report.get("decision", {}).get("training_sha256") != decision["training_sha256"]
            or not (report.get("internal_evaluation") or {}).get("metrics", {}).get(selected)
            or report.get("decision", {}).get("settings") != decision.get("settings")
            or report.get("decision", {}).get("code") != decision.get("code")
            or report.get("decision", {}).get("versions") != decision.get("versions")
            or report.get("decision", {}).get("params") != decision["params"]
            or contract.get("validation_report_sha256") != _digest(report)):
        raise ValueError("Falta evidencia coherente de validación interna 07")
    approval = contract.get("final_fit_authorization", {})
    if approval.get("authorized") is not True or not str(approval.get("approved_by", "")).strip():
        raise PermissionError("Falta autorización humana para el único ajuste final de 08")
    _utc(approval.get("approved_at"))
    # Enlaza los datos, parámetros, calibración y umbral autorizados, no solo una casilla.
    frozen = {key: decision[key] for key in ("model", "params", "features", "training_sha256", "settings", "versions", "code")}
    if contract.get("configuration_sha256") != _digest(frozen):
        raise ValueError("La configuración cambió después de autorizar el traspaso")
    if decision["status"] == "provisional":
        if decision.get("definitive") is not False or decision.get("definitive_approval"):
            raise PermissionError("Una decisión provisional no contiene aprobación definitiva")
    else:
        approval = decision.get("definitive_approval", {})
        if (decision.get("definitive") is not True or approval.get("approved") is not True
                or not str(approval.get("approved_by", "")).strip()
                or approval.get("validation_report_id") != report["report_id"]
                or approval.get("configuration_sha256") != contract["configuration_sha256"]):
            raise PermissionError("Estado definitivo sin aprobación externa coherente")
        _utc(approval.get("approved_at"))
    return deepcopy(decision)


def selected_validation_metrics(decision):
    validate_handoff(decision)
    selected = MODEL + (" + sigmoid" if decision["settings"][MODEL]["calibration"] == "sigmoid" else "")
    metrics = decision["handoff"]["validation_report"]["internal_evaluation"]["metrics"][selected]
    return {"internal_reserve_" + key: value for key, value in metrics.items()
            if type(value) in (int, float)}  # ni matrices ni métricas del experimento externo


def prepare_handoff(decision, report, *, authorize_final_fit=False, approved_by=None, approved_at=None):
    if authorize_final_fit is not True:
        raise PermissionError("Preparar el traspaso requiere autorización explícita de ajuste final")
    if decision.get("status") != "provisional":
        raise ValueError("El traspaso parte de una decisión provisional congelada")
    result = deepcopy(decision)
    result["handoff"] = {"format_version": 1, "final_fit_owner": "08", "expected_training_shape": [24000, 25],
        "input_columns": list(PREDICTORS), "validation_report": deepcopy(report),
        "validation_report_sha256": _digest(report),
        "final_fit_authorization": {"authorized": True, "approved_by": approved_by, "approved_at": approved_at}}
    result["handoff"]["configuration_sha256"] = _digest({key: result[key] for key in
        ("model", "params", "features", "training_sha256", "settings", "versions", "code")})
    return validate_handoff(result)  # sigue PROVISIONAL, sin aprobación definitiva


def approve_for_inference(decision, *, approved=False, approval=None):
    """Solo una aprobación externa explícita puede cambiar estado; nunca se llama por métricas."""
    if approved is not True:
        raise PermissionError("La aprobación definitiva debe ser explícita")
    result = validate_handoff(decision)
    record = deepcopy(approval or {})
    if (record.get("approved") is not True or not str(record.get("approved_by", "")).strip()
            or record.get("validation_report_id") != result["handoff"]["validation_report"]["report_id"]
            or record.get("configuration_sha256") != result["handoff"]["configuration_sha256"]):
        raise PermissionError("Falta aprobación definitiva ligada al informe y configuración revisados")
    _utc(record.get("approved_at"))
    result.update(status="approved_definitive", definitive=True, definitive_approval=record)
    return result


def write_handoff(decision, destination):
    validate_handoff(decision)
    payload = _bytes(decision)
    path = Path(destination)
    if path.suffix != ".json":
        raise ValueError("El contrato se guarda exclusivamente como JSON")
    with path.open("xb") as stream:
        stream.write(payload)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}


def read_handoff(path, *, trusted=False, expected_sha256=None):
    if trusted is not True:
        raise PermissionError("Confirmar procedencia del contrato JSON")
    path = Path(path)
    if path.suffix != ".json" or not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("Se requiere contrato JSON y huella externa SHA-256")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("El contrato fue modificado o su huella no coincide")
    return validate_handoff(json.loads(payload))
