"""Inferencia 09 sobre paquetes 08; sin ajuste, acceso a datasets ni registro remoto."""
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
import json

import numpy as np
import pandas as pd

from proyecto.inference.predict import InputValidationError, predict_with_scores, validate_inputs
from proyecto.models.persistence import (load_bundle, require_approved_model,
                                        validate_input_schema)


@dataclass(frozen=True)
class InferenceControls:
    diagnostic_mode: bool = False
    allow_real_inference: bool = False
    allow_real_publication: bool = False
    allow_training: bool = False
    allow_test_access: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in self.__dict__.values()):
            raise ValueError("Los controles deben ser booleanos explícitos")
        if self.allow_training or self.allow_test_access:
            raise PermissionError("La etapa 09 no admite entrenamiento ni acceso al conjunto de prueba")
        if self.diagnostic_mode and (self.allow_real_inference or self.allow_real_publication):
            raise ValueError("No combinar diagnóstico sintético con permisos reales")


def _threshold(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("El umbral debe ser un número finito entre 0 y 1, excluidos los extremos")
    return float(value)


@dataclass(frozen=True)
class InferenceSession:
    bundle: object
    controls: InferenceControls
    threshold: float
    threshold_source: str


def _authorize(bundle, controls):
    if controls.diagnostic_mode:
        if bundle.metadata["data_scope"] != "synthetic" or bundle.metadata["status"] != "provisional":
            raise PermissionError("Diagnóstico: solo paquetes sintéticos provisionales")
    else:
        if not controls.allow_real_inference:
            raise PermissionError("Inferencia real desactivada")
        require_approved_model(bundle.metadata)


def open_session(directory, *, trusted=False, manifest_sha256=None,
                 controls=InferenceControls(), synthetic_threshold=None):
    """No obtiene aprobación ni elige umbrales. Requiere huella externa también en diagnóstico."""
    if not controls.diagnostic_mode and not controls.allow_real_inference:
        raise PermissionError("Seleccione diagnóstico sintético o autorice inferencia real")
    if trusted is not True:
        raise PermissionError("Confirme la procedencia confiable del paquete Joblib")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise ValueError("Se requiere la huella SHA-256 externa del manifiesto de etapa 08")
    bundle = load_bundle(directory, trusted=True, expected_manifest_sha256=manifest_sha256,
                         require_approved=not controls.diagnostic_mode)
    _authorize(bundle, controls)
    if controls.diagnostic_mode:
        threshold = _threshold(synthetic_threshold)
        source = "synthetic_diagnostic_only; not an approved operating threshold"
    else:
        if synthetic_threshold is not None:
            raise PermissionError("No sustituir el umbral aprobado por uno sintético")
        decision = bundle.metadata["decision"]
        choice = decision.get("settings", {}).get(decision.get("model"), {})
        threshold = _threshold(choice.get("threshold"))
        source = "approved_frozen_decision.settings"
    return InferenceSession(bundle, controls, threshold, source)


def predict_clients(inputs, session):
    """Dict = un cliente; DataFrame = lote. Conserva orden e índice incluso repetido.

    Entradas ya limpias según 08: no coacciona texto numérico ni vuelve a ajustar
    estadísticas. Los faltantes presentes se imputan en el pipeline guardado.
    """
    _authorize(session.bundle, session.controls)
    threshold = _threshold(session.threshold)
    if not session.controls.diagnostic_mode:
        decision = session.bundle.metadata["decision"]
        approved_threshold = decision.get("settings", {}).get(decision.get("model"), {}).get("threshold")
        if threshold != _threshold(approved_threshold):
            raise PermissionError("El umbral de la sesión no coincide con la decisión aprobada")
    if isinstance(inputs, dict):
        try:
            frame = pd.DataFrame([inputs])
        except (TypeError, ValueError) as error:
            raise InputValidationError("Un cliente debe ser un diccionario de valores escalares") from error
    elif isinstance(inputs, pd.DataFrame):
        frame = inputs.copy(deep=True)
    else:
        raise InputValidationError("Use un diccionario para un cliente o un DataFrame para un lote")
    if frame.columns.has_duplicates:
        raise InputValidationError("Hay nombres de columnas duplicados")
    expected = session.bundle.schema["input_columns"]
    missing = [name for name in expected if name not in frame.columns]
    extra = [name for name in frame.columns if name not in expected]
    if missing or extra:
        raise InputValidationError(f"Columnas faltantes: {missing}; columnas adicionales: {extra}")
    validate_input_schema(frame, session.bundle.schema)
    # None/pd.NA se representan como NaN numérico; no se rellenan ni se aprenden
    # estadísticas. Validar primero impide convertir silenciosamente texto a números.
    for name in frame:
        if frame[name].isna().any() or name in session.bundle.schema["categorical"]:
            frame[name] = frame[name].where(frame[name].notna(), np.nan).astype(float)
    # 1 y 1.0 son el mismo código: no advertir falsamente que es una categoría nueva.
    warning_schema = {**session.bundle.schema, "categories": {
        name: [str(float(code)) for code in known]
        for name, known in session.bundle.schema["categories"].items()}}
    warnings = validate_inputs(frame, warning_schema)
    scores = predict_with_scores(frame, session.bundle, session.bundle.schema)
    probabilities = scores[["proba_0", "proba_1"]].to_numpy()
    if (probabilities.shape != (len(frame), 2) or not np.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12)):
        raise ValueError("El pipeline devolvió probabilidades inválidas")
    metadata = session.bundle.metadata
    diagnostic = session.controls.diagnostic_mode
    classified = (probabilities[:, 1] >= threshold).astype(int)
    out = pd.DataFrame({"default_probability": probabilities[:, 1], "classification": classified,
        "result": np.where(classified == 1, "Riesgo igual o superior al umbral", "Riesgo por debajo del umbral"),
        "threshold": threshold, "threshold_source": session.threshold_source,
        "model_name": metadata["model_name"], "model_version": metadata["model_version"],
        "model_status": metadata["status"], "artifact_sha256": metadata["artifact_sha256"],
        "diagnostic_only": diagnostic}, index=frame.index)
    out.attrs.update(warnings=warnings, artifact_id=metadata["artifact_id"],
        manifest_sha256=session.bundle.manifest_sha256,
        scope="synthetic_diagnostic" if diagnostic else "authorized_real_inference",
        interpretation="Probabilidad estimada, no certeza ni explicación causal; no constituye aprobación de crédito")
    return out


def publish_predictions(result, session, destination):
    """Exportación explícita a un archivo nuevo; no hay publicación automática ni remota."""
    if not session.controls.allow_real_publication or session.controls.diagnostic_mode:
        raise PermissionError("Publicación de predicciones reales no autorizada")
    _authorize(session.bundle, session.controls)
    if (result.attrs.get("artifact_id") != session.bundle.metadata["artifact_id"]
            or result.attrs.get("scope") != "authorized_real_inference"
            or result.attrs.get("manifest_sha256") != session.bundle.manifest_sha256):
        raise ValueError("El resultado no corresponde a esta sesión autorizada")
    payload = {"metadata": result.attrs, "rows": result.reset_index(drop=True).to_dict(orient="records")}
    # No crear carpetas ni sobrescribir informes. La aplicación elige un destino autorizado.
    with Path(destination).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)


def synthetic_clients():
    """Ejemplo determinista de entradas, sin leer datos ni ajustar un modelo."""
    row = {"LIMIT_BAL": 100000., "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 35.}
    for name in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        row[name] = 0
    for i in range(1, 7):
        row[f"BILL_AMT{i}"] = 20000.
        row[f"PAY_AMT{i}"] = 2000.
    second = {**row, "SEX": 2, "AGE": 48., "PAY_0": 2, "BILL_AMT1": -100.}
    return pd.DataFrame([row, second], index=["sintetico_A", "sintetico_B"])
