"""Lectura y presentación de los resultados guardados de 07-09 (para notebooks e informes; no entrena ni abre la prueba)."""
import json
from pathlib import Path

import pandas as pd

METRIC_COLUMNS = ["roc_auc", "avg_precision", "brier", "precision", "recall", "flagged_share", "penalty"]
SOURCES = (("Validación cruzada", "oof_metrics"), ("Reserva interna", "reserve_metrics"), ("Prueba (umbral 1/6)", "test_metrics"))


def load_results(root):
    """Devuelve (resultados de 07-09, recibo del registro) guardados en reports/."""
    root = Path(root)
    results = json.loads((root / "reports/resultados_07_09.json").read_text(encoding="utf-8"))
    registry = json.loads((root / "reports/registro_08.json").read_text(encoding="utf-8"))
    return results, registry


def metrics_table(results):
    """Tabla (fuente, modelo) x métricas con validación cruzada, reserva y prueba."""
    rows = {}
    for label, key in SOURCES:
        for model, values in (results.get(key) or {}).items():
            rows[(label, model)] = {column: values[column] for column in METRIC_COLUMNS}
    return pd.DataFrame(rows).T.round(4)


def paired_difference(results, reference="Logística E"):
    """Diferencia pareada (modelo de referencia menos LightGBM) en la prueba, con su intervalo."""
    table = pd.DataFrame(results["test_comparisons"][reference]).T
    return table[["difference", "low", "high", "valid_draws"]].round(4)


def demo_cases(probabilities, threshold):
    """Índices de un cliente normal (menor riesgo), uno límite (el primero que alcanza el umbral) y uno de alto riesgo."""
    if not 0 < threshold < 1:
        raise ValueError("El umbral debe estar entre 0 y 1")
    above = probabilities[probabilities >= threshold]
    if above.empty:
        raise ValueError("Ningún cliente alcanza el umbral")
    return {"Normal (menor riesgo)": probabilities.idxmin(), "Límite (justo en el umbral)": above.idxmin(),
            "Alto riesgo": probabilities.idxmax()}


def invalid_age_example(row, age=5):
    """Copia de un cliente con una edad imposible, para comprobar que la app la rechaza."""
    invalid = row.copy()
    invalid["AGE"] = age
    return invalid
