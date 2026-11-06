"""Inferencia: validar entradas y predecir clase y probabilidades.

Contrato (el mismo que el del ejemplo del curso):
  - Entrada: una tabla con las columnas CRUDAS que una persona conoce al decidir (las del esquema).
  - Salida: la clase decidida, la probabilidad de cada clase y una señal de "revisar" cuando el modelo duda.
  - Nunca devuelve una predicción silenciosa con datos imposibles: lanza `InputValidationError`.

Uso:

    from proyecto.inference.predict import load_model, predict_with_scores
    from proyecto.inference.schema import load_schema
    model, schema = load_model("models/model.joblib"), load_schema("models/input_schema.json")
    resultado = predict_with_scores(filas_nuevas, model, schema)

Funciona con cualquier clasificador de scikit-learn (o un Pipeline) que tenga `predict_proba`.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class InputValidationError(ValueError):
    """Los datos de entrada no son válidos; el mensaje dice qué corregir."""


def load_model(path: str | Path):
    """Carga el modelo exportado (un Pipeline completo, con el preprocesamiento adentro)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta antes el notebook 09_inferencia para exportarlo.")
    return joblib.load(path)


def validate_inputs(df: pd.DataFrame, schema: dict) -> list[str]:
    """Valida columnas, tipos y rangos. Lanza InputValidationError si hay errores; devuelve advertencias."""
    numeric, categorical = schema["numeric"], schema["categorical"]
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise InputValidationError(f"Faltan columnas: {', '.join(missing)}")

    errors, warnings = [], []
    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce")
        if (values.isna() & df[col].notna()).any():
            errors.append(f"{col}: hay valores que no son números")
        if col in schema.get("valid_range", {}):
            low, high = schema["valid_range"][col]
            if (values.notna() & ~values.between(low, high)).any():
                errors.append(f"{col}: debe estar entre {low:g} y {high:g}")
    if errors:
        raise InputValidationError("; ".join(errors))

    for col, known in schema.get("categories", {}).items():
        unseen = set(df[col].dropna().astype(str)) - set(known)
        if unseen:
            warnings.append(f"{col}: categoría no vista en el entrenamiento ({', '.join(sorted(unseen))})")
    for col, (low, high) in schema.get("observed_range", {}).items():
        values = pd.to_numeric(df[col], errors="coerce")
        if ((values < low) | (values > high)).any():
            warnings.append(f"{col}: fuera del rango visto en el entrenamiento ({low:g} a {high:g})")
    return warnings


def decide_min_cost(proba: np.ndarray, cost_matrix) -> np.ndarray:
    """Elige la clase de menor costo esperado: argmin_j sum_i p_i * costo(i, j)."""
    return np.argmin(proba @ np.asarray(cost_matrix, dtype=float), axis=1)


def decide_by_threshold(proba: np.ndarray, class_names: list[str], threshold: float, positive_class: str) -> np.ndarray:
    """Decisión binaria con umbral: elige la clase positiva si su probabilidad llega al umbral.

    Con un umbral menor que 0,5 el modelo marca más clientes que con «la clase más probable»;
    es la regla con la que se validó el modelo (no un costo medido, solo un punto de trabajo).
    """
    if len(class_names) != 2 or positive_class not in class_names:
        raise ValueError("El umbral de decisión solo aplica a dos clases y requiere positive_class")
    if isinstance(threshold, bool) or not 0 < float(threshold) < 1:
        raise ValueError("decision_threshold debe estar entre 0 y 1 (sin los extremos)")
    positive = class_names.index(positive_class)
    flagged = proba[:, positive] >= float(threshold)
    return np.where(flagged, positive, 1 - positive)


def _ordered_proba(model, X: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    """Probabilidades en el orden de class_names (reordena si el modelo guarda sus clases con otro orden)."""
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):  # Pipeline
        classes = getattr(model.steps[-1][1], "classes_", None)
    if classes is not None:
        as_text = [str(c) for c in classes]
        if set(as_text) == set(class_names) and as_text != class_names:
            proba = proba[:, [as_text.index(name) for name in class_names]]
    return proba


def predict_with_scores(df: pd.DataFrame, model, schema: dict) -> pd.DataFrame:
    """Predice. Una fila de salida por fila de entrada.

    Columnas: decision, most_likely, proba_<clase>, needs_review y (si el esquema trae matriz de costos)
    expected_cost. Las advertencias de validación quedan en `resultado.attrs["warnings"]`.
    """
    warnings = validate_inputs(df, schema)
    class_names = schema["class_names"]
    proba = _ordered_proba(model, df, class_names)
    most_likely = proba.argmax(axis=1)

    cost = schema.get("cost_matrix")
    if cost is not None:
        decision = decide_min_cost(proba, cost)
    elif schema.get("decision_threshold") is not None:
        decision = decide_by_threshold(proba, class_names, schema["decision_threshold"], schema["positive_class"])
    else:
        decision = most_likely

    out = pd.DataFrame(proba, columns=[f"proba_{c}" for c in class_names], index=df.index)
    out.insert(0, "decision", [class_names[i] for i in decision])
    out["most_likely"] = [class_names[i] for i in most_likely]
    if cost is not None:
        out["expected_cost"] = (proba @ np.asarray(cost, dtype=float))[np.arange(len(df)), decision]
    out["needs_review"] = (proba.max(axis=1) < schema.get("review_threshold", 0.55)) | (decision != most_likely)
    out.attrs["warnings"] = warnings
    return out
