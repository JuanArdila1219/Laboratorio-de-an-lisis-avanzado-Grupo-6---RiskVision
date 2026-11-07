"""Adaptador de la app para el artefacto exportado por Andrés; nunca entrena."""
from importlib.metadata import version
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from proyecto.data.cleaning import _validate_predictors, NON_NEGATIVE
from proyecto.inference.predict import InputValidationError, predict_with_scores


def operating_threshold(schema):
    value = schema.get("decision_threshold")
    if type(value) not in (float, int) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("El esquema debe declarar un umbral operativo válido")
    if schema.get("positive_class") != "1" or schema.get("class_names") != ["0", "1"] or schema.get("cost_matrix") is not None:
        raise ValueError("Se requiere clasificación binaria con umbral, sin otra regla de costos activa")
    return float(value)


def load_assets(model_path, schema_path, *, trusted=False):
    if trusted is not True:
        raise PermissionError("Solo cargar el modelo de procedencia confiable del repositorio revisado")
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    operating_threshold(schema)
    artifact = schema["artifact"]
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("El modelo exportado requiere Python 3.12")
    differences = {name: (expected, version(name)) for name, expected in artifact["dependencies"].items()
                   if version(name) != expected}
    if differences:
        raise RuntimeError(f"Dependencias distintas a las del modelo: {differences}")
    payload = Path(model_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
        raise ValueError("SHA-256 del modelo distinto del artefacto revisado")
    model = joblib.load(BytesIO(payload))
    if list(model.feature_names_in_) != schema["input_columns"] or list(model.classes_) != [0, 1]:
        raise ValueError("Modelo y esquema de entrada incompatibles")
    return model, schema


def prepare_inputs(frame, schema):
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.has_duplicates:
        raise InputValidationError("La entrada debe ser una tabla no vacía y sin columnas duplicadas")
    expected = schema["input_columns"]
    missing, extra = set(expected) - set(frame.columns), set(frame.columns) - set(expected)
    if missing or extra:
        raise InputValidationError(f"Columnas faltantes: {sorted(missing)}; adicionales: {sorted(extra)}")
    frame = frame.loc[:, expected].copy()
    # El formulario entrega etiquetas textuales de códigos numéricos, no categorías de texto.
    for name, domain in schema["categorical_domains"].items():
        values = []
        for value in frame[name]:
            if pd.isna(value):
                values.append(np.nan)
                continue
            try:
                code = float(value)
            except (TypeError, ValueError) as error:
                raise InputValidationError(f"{name}: código numérico inválido") from error
            if isinstance(value, (bool, np.bool_)) or code not in domain:
                raise InputValidationError(f"{name}: categorías permitidas {domain}")
            values.append(code)
        frame[name] = values
    try:
        _validate_predictors(frame)
    except ValueError as error:
        raise InputValidationError(str(error)) from error
    if not frame["AGE"].dropna().between(18, 100).all():
        raise InputValidationError("AGE: debe estar entre 18 y 100")
    if (frame["LIMIT_BAL"].dropna() < 1).any():
        raise InputValidationError("LIMIT_BAL: debe ser positivo")
    if any((frame[name].dropna() < 0).any() for name in NON_NEGATIVE):
        raise InputValidationError("PAY_AMT: los pagos no pueden ser negativos")
    for name in frame:
        if frame[name].isna().any():
            frame[name] = frame[name].where(frame[name].notna(), np.nan).astype(float)
    return frame


def score_clients(frame, model, schema):
    operating_threshold(schema)
    prepared = prepare_inputs(frame, schema)
    warning_schema = {**schema, "categories": {name: [str(float(code)) for code in known]
        for name, known in schema["categories"].items()}}
    result = predict_with_scores(prepared, model, warning_schema)
    probability = result[["proba_0", "proba_1"]].to_numpy()
    if (not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all()
            or not np.allclose(probability.sum(axis=1), 1)):
        raise ValueError("Probabilidades inválidas del modelo")
    result.attrs.update(operating_threshold=operating_threshold(schema),
        threshold_author=schema["decision_provenance"]["author"],
        artifact_sha256=schema["artifact"]["sha256"], approval_status=schema["artifact"]["approval_status"])
    return result
