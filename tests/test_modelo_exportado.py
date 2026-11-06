"""Comprueba que models/model.joblib + models/input_schema.json (lo que carga la app) son coherentes."""
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from proyecto.data.split import make_xy
from proyecto.inference.predict import predict_with_scores

ROOT = Path(__file__).resolve().parents[1]
MODEL, SCHEMA, TRAIN = ROOT / "models/model.joblib", ROOT / "models/input_schema.json", ROOT / "data/processed/train.joblib"
pytestmark = pytest.mark.skipif(not (MODEL.exists() and SCHEMA.exists() and TRAIN.exists()),
                                reason="faltan el modelo exportado, su esquema o los datos procesados")


@pytest.fixture(scope="module")
def scored():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    X, _ = make_xy(joblib.load(TRAIN))
    sample = X.sample(300, random_state=0)
    return schema, predict_with_scores(sample, joblib.load(MODEL), schema)


def test_schema_declares_the_working_threshold(scored):
    schema, _ = scored
    assert schema["positive_class"] == "1" and 0 < schema["decision_threshold"] < 0.5


def test_app_decision_matches_the_declared_threshold(scored):
    schema, result = scored
    expected = np.where(result["proba_1"] >= schema["decision_threshold"], "1", "0")
    assert (result["decision"].to_numpy() == expected).all()


def test_threshold_flags_more_clients_than_most_likely_class(scored):
    _, result = scored
    assert (result["decision"] == "1").sum() > (result["most_likely"] == "1").sum()
