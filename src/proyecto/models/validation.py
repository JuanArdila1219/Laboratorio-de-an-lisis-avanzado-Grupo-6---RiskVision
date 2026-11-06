"""Etapa 07: validación interna y evaluación complementaria con autorización.

Una fila es un cliente único. La reserva y la prueba original ya estuvieron
expuestas: ninguna métrica o intervalo de este módulo elimina ese antecedente.
No ejecuta nada al importarse, ni selecciona un modelo definitivo.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
import hashlib
from io import BytesIO
import json
from numbers import Real
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from proyecto.config import ID_COL, TARGET
from proyecto.data.cleaning import REQUIRED_COLUMNS, _validate_training_frame
from proyecto.data.split import assert_disjoint_ids, assert_unique_ids, make_xy, split_stratified
from proyecto.features.build import PACKAGE_E
from proyecto.models.candidates import LGBM_PARAMS, make_lgbm, make_logistic
from proyecto.models.evaluate import (THEORETICAL_THRESHOLD, calibration_table,
                                     expected_calibration_error, relative_penalty, technical_metrics)

ORIGINAL = "LightGBM original"
REFERENCE = "Logística E"
OPTIMIZED = "LightGBM optimizado"
LIMITATIONS = [
    "Prueba original y reserva expuestas previamente: no evaluación independiente",
    "Selección previa de variables/modelos y posible optimismo de selección",
    "Intervalos condicionados a los modelos ajustados; no incluyen repetir selección o ajuste",
    "Penalización 5:1 hipotética y capacidad del 20 % supuesta; no ahorro económico demostrado",
    "Interpretación asociativa del modelo, no causal ni recomendación automática de negar crédito",
]


@dataclass(frozen=True)
class ExecutionControls:
    fit_real: bool = False
    calibrate_real: bool = False
    evaluate_test: bool = False
    export_definitive: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("Los permisos deben ser booleanos explícitos")


def _require_fit(controls, calibrated=False):
    if not controls.fit_real:
        raise PermissionError("Ajuste bloqueado: ALLOW_REAL_FIT=False")
    if calibrated and not controls.calibrate_real:
        raise PermissionError("Calibración bloqueada: ALLOW_REAL_CALIBRATION=False")


def _training_frame(frame):
    if frame.attrs.get("role") not in (None, "training", "training_development", "synthetic_training"):
        raise PermissionError("La reserva o prueba no pueden utilizarse para ajustar o seleccionar")
    _validate_training_frame(frame)
    if set(frame.columns) != set(REQUIRED_COLUMNS):
        raise ValueError("Se requiere el esquema original de 25 columnas, sin variables precalculadas")


def load_training(path, *, expected_sha256, expected_rows=24000):
    path = Path(path)
    if path.name != "train.joblib":
        raise ValueError("Solo se admite train.joblib como entrenamiento")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("La huella del entrenamiento no coincide con la aprobada")
    frame = joblib.load(BytesIO(payload))
    _training_frame(frame)
    if frame.shape != (expected_rows, 25):
        raise ValueError("Dimensiones inesperadas del entrenamiento")
    frame.attrs.update(role="training", source_sha256=digest)
    return frame


def provisional_decision(*, revision, versions, training_sha256):
    return {"status": "provisional", "definitive": False, "model": ORIGINAL,
            "features": list(PACKAGE_E), "params": deepcopy(LGBM_PARAMS),
            "created_at": datetime.now(timezone.utc).isoformat(), "code": dict(revision),
            "versions": dict(versions), "training_sha256": training_sha256,
            "parameter_source": "src/proyecto/models/candidates.py:LGBM_PARAMS",
            "own_real_optimization_executed": False,
            "antecedent": {"branch": "origin/Estefania", "commit": "e4067168e4f75dfe3c4f9f5e83cf20df797775c5",
                           "conclusion": "Conservar parámetros originales; experimento distinto, sin atribución propia"},
            "calibration": "pending", "threshold": "pending", "limitations": list(LIMITATIONS)}


def candidate_factories(*, optimized_evidence=None, approve_optimized=False, training_sha256=None):
    factories = {ORIGINAL: partial(make_lgbm, PACKAGE_E), REFERENCE: partial(make_logistic, PACKAGE_E)}
    if optimized_evidence is None:
        return factories
    if approve_optimized is not True:
        raise PermissionError("Los parámetros optimizados necesitan aprobación explícita")
    # Solo el formato completo de nuestro protocolo, no el JSON seleccionado de otro experimento.
    from proyecto.models.tuning import SearchSpace, TuningConfig, make_trial_pipeline
    evidence = optimized_evidence
    if not isinstance(evidence, dict):
        raise ValueError("La evidencia debe ser un informe estructurado")
    expected_space = json.loads(json.dumps(asdict(SearchSpace())))
    valid = (evidence.get("metric") == "roc_auc" and evidence.get("direction") == "MAXIMIZE"
             and evidence.get("features") == list(PACKAGE_E)
             and training_sha256 is not None and evidence.get("training_sha256") == training_sha256
             and evidence.get("config") == asdict(TuningConfig())
             and json.loads(json.dumps(evidence.get("search_space"))) == expected_space
             and evidence.get("reserve", {}).get("seed") == 7
             and evidence.get("reserve", {}).get("share") == .25
             and evidence.get("code", {}).get("branch") == "revision-h2-riskvision"
             and bool(evidence.get("code", {}).get("commit"))
             and isinstance(evidence.get("best_cv_auc"), (int, float))
             and 0 <= evidence["best_cv_auc"] <= 1)
    if not valid:
        raise ValueError("Evidencia incompleta o incompatible con el protocolo aprobado")
    params = dict(evidence.get("best_params_candidate", {}))
    trial = next((trial for trial in evidence.get("trials", []) if trial.get("number") == evidence.get("best_trial")
                  and trial.get("state") == "COMPLETE"), None)
    if (not evidence.get("report_id") or trial is None or trial.get("params") != params
            or trial.get("roc_auc_cv") != evidence["best_cv_auc"]):
        raise ValueError("Falta trazabilidad coherente del mejor ensayo completado")
    make_trial_pipeline(params)  # valida sin ajustar
    factories[OPTIMIZED] = partial(make_trial_pipeline, params, seed=42, model_n_jobs=1)
    return factories


def _vectors(y, p, ids=None):
    labels, probability = np.asarray(y), np.asarray(p, dtype=float)
    if labels.ndim != 1 or probability.ndim != 1 or len(labels) != len(probability) or not len(labels):
        raise ValueError("Vectores vacíos o de dimensiones incompatibles")
    if not pd.Series(labels).isin([0, 1]).all():
        raise ValueError("Objetivo binario sin faltantes requerido")
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Probabilidades finitas entre 0 y 1 requeridas")
    if ids is not None:
        if np.asarray(ids).ndim != 1 or len(ids) != len(labels):
            raise ValueError("Identificadores incompatibles")
        assert_unique_ids(pd.DataFrame({ID_COL: np.asarray(ids)}))
    return labels.astype(int), probability


def _threshold(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("El umbral debe estar entre 0 y 1")
    return float(value)


def metrics(y, p, threshold=THEORETICAL_THRESHOLD):
    y, p = _vectors(y, p)
    threshold = _threshold(threshold)
    flagged = p >= threshold
    both = len(np.unique(y)) == 2
    if both:
        result = technical_metrics(y, p, threshold=threshold)
    else:
        # AUC/AP de un segmento con una sola clase no fundamentan discriminación.
        matrix = confusion_matrix(y, flagged, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        result = {"roc_auc": None, "avg_precision": None,
                  "precision_1": float(tp / (tp + fp)) if tp + fp else 0.,
                  "recall_1": float(tp / (tp + fn)) if tp + fn else 0.,
                  "brier": float(brier_score_loss(y, p)),
                  "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
                  "penalty": relative_penalty(y, p, threshold), "flagged_share": float(flagged.mean()),
                  "recall_top": None}
    return {**result, "precision": result["precision_1"], "recall": result["recall_1"],
            "f1": float(f1_score(y, flagged, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, flagged, labels=[0, 1]).tolist(),
            "threshold": threshold, "n": len(y), "positive_rate": float(y.mean()), "single_class": not both}


def fit_training_candidate(frame, factory, *, controls, calibrated=False, seed=42):
    _require_fit(controls, calibrated)
    _training_frame(frame)
    X, y = make_xy(frame)
    estimator = clone(factory())
    if calibrated:
        if y.value_counts().min() < 3:
            raise ValueError("Faltan clientes por clase para calibrar con tres pliegues")
        estimator = CalibratedClassifierCV(estimator, method="sigmoid",
            cv=StratifiedKFold(3, shuffle=True, random_state=seed), ensemble=False)
    return estimator.fit(X, y)


def crossfit_probabilities(frame, factory, *, controls, calibrated=False, n_splits=5, seed=42):
    """Una predicción OOF por cliente; calibrador dentro del entrenamiento de cada pliegue."""
    _require_fit(controls, calibrated)
    _training_frame(frame)
    X, y = make_xy(frame)
    if y.nunique() != 2 or y.value_counts().min() < n_splits:
        raise ValueError("Faltan clientes de ambas clases para los pliegues")
    proba, fold_number = np.full(len(frame), np.nan), np.full(len(frame), -1)
    audits = []
    for fold, (train_idx, valid_idx) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)):
        fitted = fit_training_candidate(frame.iloc[train_idx], factory, controls=controls,
                                        calibrated=calibrated, seed=seed)
        proba[valid_idx] = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        fold_number[valid_idx] = fold
        audits.append({"train_ids": frame.iloc[train_idx][ID_COL].tolist(),
                       "validation_ids": frame.iloc[valid_idx][ID_COL].tolist()})
    result = pd.DataFrame({ID_COL: frame[ID_COL].to_numpy(), "y_true": y.to_numpy(),
                           "proba": proba, "fold": fold_number})
    result.attrs.update(role="training_oof", folds=audits)
    return result


def threshold_analysis(y, p, thresholds=None):
    values = sorted(set([*np.linspace(.05, .95, 19), THEORETICAL_THRESHOLD] if thresholds is None else thresholds))
    if not values:
        raise ValueError("Se requiere al menos un umbral")
    return pd.DataFrame([metrics(y, p, _threshold(value)) for value in values])


def select_threshold(oof, thresholds=None):
    if oof.attrs.get("role") != "training_oof":
        raise PermissionError("Solo se seleccionan umbrales con predicciones OOF del entrenamiento")
    _vectors(oof["y_true"], oof["proba"], oof[ID_COL])
    table = threshold_analysis(oof["y_true"], oof["proba"], thresholds)
    # Criterio propuesto previo a resultados: penalización, carga y umbral más alto.
    best = table.sort_values(["penalty", "flagged_share", "threshold"], ascending=[True, True, False]).iloc[0]
    return {"threshold": float(best["threshold"]), "criterion": "5:1 hipotético; desempate por menor carga y mayor umbral",
            "scope": "training_oof", "status": "recommendation_pending_review"}


def calibration_diagnostics(y, p, n_bins=10):
    y, p = _vectors(y, p)
    table = calibration_table(y, p, n_bins=min(n_bins, len(y)))
    return {"curve": table, "ece": expected_calibration_error(table),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))}


def recommend_calibration(raw_oof, calibrated_oof):
    for frame in (raw_oof, calibrated_oof):
        if frame.attrs.get("role") != "training_oof":
            raise PermissionError("La recomendación de calibración solo utiliza entrenamiento OOF")
    if not np.array_equal(raw_oof[[ID_COL, "y_true"]], calibrated_oof[[ID_COL, "y_true"]]):
        raise ValueError("Los clientes y objetivos de la comparación no coinciden")
    raw, calibrated = (metrics(frame["y_true"], frame["proba"]) for frame in (raw_oof, calibrated_oof))
    recommend = (calibrated["brier"] < raw["brier"] and calibrated["log_loss"] < raw["log_loss"]
                 and calibrated["roc_auc"] >= raw["roc_auc"])
    return {"recommend_sigmoid": bool(recommend), "status": "recommendation_pending_review",
            "criterion": "Brier y log loss menores, sin descenso del AUC OOF; no selección por reserva/prueba"}


def paired_bootstrap(y, reference, variant, ids, *, metric="roc_auc", n_boot=1000, seed=42,
                     reference_threshold=THEORETICAL_THRESHOLD, variant_threshold=THEORETICAL_THRESHOLD):
    """AUC recalculado por remuestra o penalización media; nunca AUC por fila."""
    y, reference = _vectors(y, reference, ids)
    _, variant = _vectors(y, variant)
    if metric not in {"roc_auc", "penalty"} or type(n_boot) is not int or n_boot < 2:
        raise ValueError("Métrica o presupuesto de bootstrap inválido")
    if metric == "roc_auc" and len(np.unique(y)) != 2:
        raise ValueError("AUC requiere ambas clases")
    reference_threshold, variant_threshold = _threshold(reference_threshold), _threshold(variant_threshold)
    from sklearn.metrics import roc_auc_score
    def score(labels, prediction, threshold):
        return roc_auc_score(labels, prediction) if metric == "roc_auc" else relative_penalty(labels, prediction, threshold)
    reference_score, variant_score = score(y, reference, reference_threshold), score(y, variant, variant_threshold)
    rng, differences = np.random.default_rng(seed), []
    for _ in range(n_boot):
        indices = rng.integers(0, len(y), len(y))
        if metric == "roc_auc" and len(np.unique(y[indices])) < 2:
            continue
        differences.append(score(y[indices], variant[indices], variant_threshold)
                           - score(y[indices], reference[indices], reference_threshold))
    if len(differences) < 2:
        raise ValueError("Remuestras válidas insuficientes")
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"metric": metric, "difference": float(variant_score - reference_score),
            "low": float(low), "high": float(high), "valid_draws": len(differences), "requested_draws": n_boot,
            "sign": "variant_minus_reference", "independent_evaluation": False}


def evaluate_internal(frame, *, controls, optimized_evidence=None, approve_optimized=False,
                      n_splits=5, seed=42, reserve_seed=7, n_boot=1000):
    _require_fit(controls)
    _training_frame(frame)
    development, reserve = split_stratified(frame, test_size=.25, seed=reserve_seed)
    development.attrs["role"], reserve.attrs["role"] = "training_development", "internal_reserve"
    assert_disjoint_ids(development, reserve)
    factories = candidate_factories(optimized_evidence=optimized_evidence, approve_optimized=approve_optimized,
                                   training_sha256=frame.attrs.get("source_sha256"))
    X_reserve, y_reserve = make_xy(reserve)
    result = {"development": development, "reserve": reserve, "models": {}, "probabilities": {},
              "thresholds": {}, "oof": {}, "calibration": {}, "recommendations": {}, "comparisons": {}}
    rows, oof_rows = {}, {}
    for name, factory in factories.items():
        for calibrated in ([False, True] if controls.calibrate_real else [False]):
            label = name + (" + sigmoid" if calibrated else "")
            oof = crossfit_probabilities(development, factory, controls=controls, calibrated=calibrated,
                                         n_splits=n_splits, seed=seed)
            proposed = select_threshold(oof)
            threshold = proposed["threshold"]
            model = fit_training_candidate(development, factory, controls=controls, calibrated=calibrated, seed=seed)
            probability = model.predict_proba(X_reserve)[:, 1]
            result["models"][label], result["probabilities"][label] = model, probability
            result["thresholds"][label], result["oof"][label] = proposed, oof
            result["calibration"][label] = calibration_diagnostics(oof["y_true"], oof["proba"])
            rows[label], oof_rows[label] = metrics(y_reserve, probability, threshold), metrics(oof["y_true"], oof["proba"], threshold)
        if controls.calibrate_real:
            result["recommendations"][name] = recommend_calibration(result["oof"][name], result["oof"][name + " + sigmoid"])
    result["metrics"], result["oof_metrics"] = pd.DataFrame.from_dict(rows, orient="index"), pd.DataFrame.from_dict(oof_rows, orient="index")
    for name, probability in result["probabilities"].items():
        if name != ORIGINAL:
            result["comparisons"][name] = {metric: paired_bootstrap(y_reserve, result["probabilities"][ORIGINAL], probability,
                reserve[ID_COL], metric=metric, n_boot=n_boot, seed=seed,
                reference_threshold=result["thresholds"][ORIGINAL]["threshold"],
                variant_threshold=result["thresholds"][name]["threshold"]) for metric in ("roc_auc", "penalty")}
    return result


def freeze_for_evaluation(decision, settings, *, approved=False, candidates=(ORIGINAL, REFERENCE)):
    if approved is not True:
        raise PermissionError("Congelar decisiones requiere aprobación explícita")
    if settings is None or set(settings) != set(candidates):
        raise ValueError("Define calibración y umbral de todos los candidatos antes de evaluar la prueba")
    for choice in settings.values():
        if choice.get("calibration") not in {"none", "sigmoid"}:
            raise ValueError("Calibración pendiente o inválida")
        _threshold(choice.get("threshold"))
    frozen = deepcopy(decision)
    frozen.update(frozen_for_evaluation=True, settings=deepcopy(settings),
                  calibration={name: choice["calibration"] for name, choice in settings.items()},
                  threshold={name: choice["threshold"] for name, choice in settings.items()},
                  frozen_at=datetime.now(timezone.utc).isoformat())
    return frozen  # sigue siendo provisional; no equivale a aprobación definitiva


def load_complementary_test(path, *, controls, decision, training_ids, expected_rows=6000):
    if not controls.evaluate_test:
        raise PermissionError("Lectura de prueba bloqueada: ALLOW_TEST_EVALUATION=False")
    if not decision or not decision.get("frozen_for_evaluation"):
        raise PermissionError("Faltan decisiones congeladas y aprobadas")
    path = Path(path)
    if path.name != "test.joblib":
        raise ValueError("Ruta de prueba no reconocida")
    payload = path.read_bytes()
    frame = joblib.load(BytesIO(payload))
    _validate_training_frame(frame)
    if frame.shape != (expected_rows, 25) or set(training_ids) & set(frame[ID_COL]):
        raise ValueError("Dimensiones inválidas o clientes compartidos con entrenamiento")
    frame.attrs.update(role="exposed_test", source_sha256=hashlib.sha256(payload).hexdigest())
    return frame


def evaluate_complementary(models, frame, *, controls, decision, n_boot=1000, seed=42):
    if not controls.evaluate_test or not decision or not decision.get("frozen_for_evaluation"):
        raise PermissionError("Evaluación complementaria bloqueada")
    if set(models) != set(decision["settings"]) or ORIGINAL not in models:
        raise ValueError("Los modelos no coinciden con las decisiones congeladas")
    X, y = make_xy(frame)
    probabilities = {name: model.predict_proba(X)[:, 1] for name, model in models.items()}
    rows = {name: metrics(y, p, decision["settings"][name]["threshold"]) for name, p in probabilities.items()}
    comparisons = {name: {metric: paired_bootstrap(y, probabilities[ORIGINAL], p, frame[ID_COL], metric=metric,
        n_boot=n_boot, seed=seed, reference_threshold=decision["settings"][ORIGINAL]["threshold"],
        variant_threshold=decision["settings"][name]["threshold"]) for metric in ("roc_auc", "penalty")}
        for name, p in probabilities.items() if name != ORIGINAL}
    return {"metrics": pd.DataFrame.from_dict(rows, orient="index"), "comparisons": comparisons,
            "probabilities": probabilities, "scope": "previously_exposed_test", "independent_evaluation": False}


def segment_metrics(frame, probability, *, threshold=THEORETICAL_THRESHOLD):
    _, y = make_xy(frame)
    _vectors(y, probability, frame[ID_COL])
    segments = {"SEX": frame["SEX"], "AGE": pd.cut(frame["AGE"], [18, 30, 45, 60, 101], right=False)}
    rows = []
    for name, groups in segments.items():
        for label, positions in groups.groupby(groups, observed=True, dropna=False).indices.items():
            rows.append({"variable": name, "segment": str(label),
                         **metrics(y.iloc[positions], np.asarray(probability)[positions], threshold)})
    return pd.DataFrame(rows)


def interpretation(estimator, frame, *, controls, n_repeats=5, seed=42):
    _require_fit(controls)
    X, y = make_xy(frame)
    result = permutation_importance(estimator, X, y, scoring="roc_auc", n_repeats=n_repeats,
                                    random_state=seed, n_jobs=1)
    return pd.DataFrame({"feature": X.columns, "auc_decrease_mean": result.importances_mean,
                         "auc_decrease_std": result.importances_std}).sort_values("auc_decrease_mean", ascending=False)


def plot_calibration(curves):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Referencia ideal")
    for name, diagnostic in curves.items():
        table = diagnostic["curve"]
        axis.plot(table["mean_predicted"], table["observed_rate"], marker="o", label=name)
    axis.set(xlabel="Probabilidad media predicha", ylabel="Frecuencia de incumplimiento", title="Calibración OOF del entrenamiento")
    axis.legend()
    return figure


def export_definitive_model(model, path, *, controls, decision):
    if not controls.export_definitive or not controls.fit_real:
        raise PermissionError("Exportación definitiva bloqueada")
    if not decision or decision.get("status") != "approved_definitive" or not decision.get("frozen_for_evaluation"):
        raise PermissionError("El modelo provisional no tiene aprobación definitiva")
    check_is_fitted(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:  # no sobrescribe modelos anteriores
        joblib.dump(model, stream)
    return path


def validation_report(decision, *, controls, internal=None, complementary=None, importance=None, segments=None):
    def table(frame):
        return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="index")
    report = {"report_id": uuid4().hex, "stage": "07", "status": "pending_review",
              "created_at": datetime.now(timezone.utc).isoformat(), "decision": deepcopy(decision),
              "controls": asdict(controls), "internal_evaluation": None, "complementary_evaluation": None,
              "interpretation": None if importance is None else importance.to_dict(orient="records"),
              "interpretation_scope": None if importance is None else "uncalibrated_original_on_internal_reserve",
              "segments": None if segments is None else segments.astype(object).where(pd.notna(segments), None).to_dict(orient="records"),
              "limitations": list(LIMITATIONS), "definitive": False}
    if internal is not None:
        report["internal_evaluation"] = {"metrics": table(internal["metrics"]), "oof_metrics": table(internal["oof_metrics"]),
            "threshold_recommendations": internal["thresholds"], "calibration_recommendations": internal["recommendations"],
            "calibration": {name: {**{key: value for key, value in diagnostic.items() if key != "curve"},
                                   "curve": diagnostic["curve"].to_dict(orient="records")}
                            for name, diagnostic in internal["calibration"].items()},
            "comparisons": internal["comparisons"], "search_clients": len(internal["development"]),
            "reserve_clients": len(internal["reserve"])}
    if complementary is not None:
        report["complementary_evaluation"] = {"metrics": table(complementary["metrics"]),
            "comparisons": complementary["comparisons"], "independent_evaluation": False}
    json.dumps(report, allow_nan=False)
    return report
