"""Pruebas del esquema y de la inferencia, con un modelo de juguete (no necesitan tus datos).

Si renombraste el paquete `proyecto`, cambia los imports de arriba. Estas pruebas son un ejemplo de
qué comprobar; cuando tengas tu modelo real, añade una prueba con una fila de tus datos.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from proyecto.inference.predict import InputValidationError, decide_min_cost, predict_with_scores, validate_inputs
from proyecto.inference.schema import build_schema, load_schema

NUMERIC, CATEGORICAL, CLASSES = ["edad", "ingreso"], ["ciudad"], ["bajo", "alto"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "edad": rng.integers(18, 70, n),
        "ingreso": rng.normal(2000, 500, n).round(1),
        "ciudad": rng.choice(["norte", "sur", "centro"], n),
    })
    df["y"] = ((df["ingreso"] > 2000) & (df["edad"] > 35)).astype(int)
    return df


@pytest.fixture(scope="module")
def model(data) -> Pipeline:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=500))]).fit(data[NUMERIC + CATEGORICAL], data["y"])


def make_schema(data, **kwargs) -> dict:
    return build_schema(data, NUMERIC, CATEGORICAL, CLASSES, valid_range={"edad": (0, 120)}, **kwargs)


def row(**overrides) -> pd.DataFrame:
    base = {"edad": 40, "ingreso": 2500.0, "ciudad": "norte"}
    base.update(overrides)
    return pd.DataFrame([base])


def test_build_schema_saves_and_reloads(data, tmp_path):
    path = tmp_path / "input_schema.json"
    schema = make_schema(data, path=path)
    assert path.exists() and load_schema(path) == json.loads(path.read_text(encoding="utf-8"))
    assert schema["numeric"] == NUMERIC and schema["class_names"] == CLASSES
    assert set(schema["categories"]["ciudad"]) == {"norte", "sur", "centro"}
    assert "edad" in schema["integer_columns"] and "ingreso" not in schema["integer_columns"]
    assert schema["observed_range"]["edad"][0] >= 18


def test_build_schema_rejects_bad_cost_matrix(data):
    with pytest.raises(ValueError, match="cost_matrix"):
        make_schema(data, cost_matrix=[[0, 1, 2], [1, 0, 2]])


def test_build_schema_rejects_missing_column(data):
    with pytest.raises(ValueError, match="Faltan columnas"):
        build_schema(data.drop(columns=["ingreso"]), NUMERIC, CATEGORICAL, CLASSES)


def test_valid_row_has_no_errors(data):
    assert validate_inputs(row(), make_schema(data)) == []


def test_missing_column_is_an_error_naming_it(data):
    with pytest.raises(InputValidationError, match="ingreso"):
        validate_inputs(row().drop(columns=["ingreso"]), make_schema(data))


def test_impossible_value_is_refused_not_predicted(data, model):
    with pytest.raises(InputValidationError, match="edad"):
        predict_with_scores(row(edad=-500), model, make_schema(data))


def test_non_numeric_value_is_an_error(data):
    with pytest.raises(InputValidationError, match="ingreso"):
        validate_inputs(row(ingreso="mucho"), make_schema(data))


def test_unseen_category_and_out_of_range_are_warnings(data):
    warnings = validate_inputs(row(ciudad="luna", ingreso=99999.0), make_schema(data))
    assert any("ciudad" in w for w in warnings) and any("ingreso" in w for w in warnings)


def test_predict_returns_probabilities_that_sum_to_one(data, model):
    result = predict_with_scores(row(), model, make_schema(data))
    assert np.isclose(result[["proba_bajo", "proba_alto"]].sum(axis=1).iloc[0], 1.0)
    assert result["decision"].iloc[0] in CLASSES and "expected_cost" not in result.columns


def test_missing_values_are_accepted_by_the_pipeline(data, model):
    result = predict_with_scores(row(ingreso=np.nan, ciudad=np.nan), model, make_schema(data))
    assert len(result) == 1


def test_cost_matrix_changes_the_decision_when_unsure():
    proba = np.array([[0.45, 0.55]])   # la clase 1 es apenas más probable
    assert decide_min_cost(proba, [[0, 1], [10, 0]])[0] == 1                 # errar la clase 1 cuesta 10: decide 1
    assert decide_min_cost(proba, [[0, 10], [1, 0]])[0] == 0                 # ahora errar la clase 0 cuesta 10: decide 0


def test_cost_matrix_adds_expected_cost_and_review_flag(data, model):
    schema = make_schema(data, cost_matrix=[[0, 1], [50, 0]])   # dejar pasar un "alto" como "bajo" cuesta 50
    result = predict_with_scores(row(edad=30, ingreso=1900.0), model, schema)
    assert "expected_cost" in result.columns and result["expected_cost"].iloc[0] >= 0


def test_app_module_imports_without_launching_streamlit():
    pytest.importorskip("streamlit")
    spec = importlib.util.spec_from_file_location("app_streamlit", ROOT / "app" / "streamlit_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)   # no ejecuta main(): solo importa
    assert callable(module.main) and module.UNKNOWN == "(no sé)"


def test_decision_threshold_flags_the_positive_class_when_probability_reaches_it():
    from proyecto.inference.predict import decide_by_threshold
    proba = np.array([[0.90, 0.10], [0.80, 0.20], [0.70, 0.30], [0.40, 0.60]])
    marked = decide_by_threshold(proba, ["0", "1"], 0.20, "1")
    assert marked.tolist() == [0, 1, 1, 1]  # 0,20 ya cuenta (>=); 0,10 no
    # con «la clase más probable» (umbral 0,5) solo se marcaría la última fila
    assert decide_by_threshold(proba, ["0", "1"], 0.5, "1").tolist() == [0, 0, 0, 1]


def test_decision_threshold_rejects_invalid_values_and_multiclass():
    from proyecto.inference.predict import decide_by_threshold
    proba = np.array([[0.5, 0.5]])
    for bad in (0, 1, -0.1, 1.5, True):
        with pytest.raises(ValueError):
            decide_by_threshold(proba, ["0", "1"], bad, "1")
    with pytest.raises(ValueError):
        decide_by_threshold(np.array([[0.2, 0.3, 0.5]]), ["a", "b", "c"], 0.2, "a")
    with pytest.raises(ValueError):
        decide_by_threshold(proba, ["0", "1"], 0.2, "x")


def test_predict_with_scores_uses_schema_threshold(data, model):
    schema = make_schema(data)
    base = predict_with_scores(row(), model, schema)
    p_high = float(base[f"proba_{CLASSES[1]}"].iloc[0])
    schema_low = {**schema, "decision_threshold": max(p_high - 0.01, 0.001), "positive_class": CLASSES[1]}
    schema_high = {**schema, "decision_threshold": min(p_high + 0.01, 0.999), "positive_class": CLASSES[1]}
    assert predict_with_scores(row(), model, schema_low)["decision"].iloc[0] == CLASSES[1]
    assert predict_with_scores(row(), model, schema_high)["decision"].iloc[0] == CLASSES[0]
