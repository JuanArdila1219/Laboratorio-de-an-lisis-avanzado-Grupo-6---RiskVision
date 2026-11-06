"""Pruebas de las piezas puras de proyecto.models.final_run (sin datos ni entrenamiento)."""
import json

import pytest

from proyecto.models import final_run as fr


def test_build_app_schema_adds_threshold_without_touching_the_original():
    original = {"input_columns": ["A"], "class_names": ["0", "1"]}
    schema = fr.build_app_schema(original, 0.2)
    assert schema["decision_threshold"] == 0.2 and schema["positive_class"] == "1"
    assert "decision_threshold" not in original  # el esquema del paquete verificado no se modifica


@pytest.mark.parametrize("bad", [0, 1, -0.2, 1.4, True])
def test_build_app_schema_rejects_invalid_threshold(bad):
    with pytest.raises(ValueError):
        fr.build_app_schema({"class_names": ["0", "1"]}, bad)


def test_export_for_app_copies_model_bytes_and_writes_threshold(tmp_path):
    bundle, out = tmp_path / "bundle", tmp_path / "models"
    bundle.mkdir(), out.mkdir()
    (bundle / "pipeline.joblib").write_bytes(b"modelo-de-prueba")
    (bundle / "input_schema.json").write_text(json.dumps({"class_names": ["0", "1"]}), encoding="utf-8")
    fr.export_for_app(bundle, out, threshold=0.25)
    assert (out / "model.joblib").read_bytes() == b"modelo-de-prueba"
    assert json.loads((out / "input_schema.json").read_text(encoding="utf-8"))["decision_threshold"] == 0.25


def test_operating_threshold_is_below_one_half():
    assert 0 < fr.OPERATING_THRESHOLD < 0.5
