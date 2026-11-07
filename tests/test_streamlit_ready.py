"""Inferencia del modelo existente: entradas artificiales, ningún ajuste ni dataset."""
import hashlib
import json
from pathlib import Path
import sys
import os
import socket
import subprocess
import time
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from proyecto.inference import app_runtime as r
from proyecto.inference.predict import InputValidationError, decide_by_threshold

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/model.joblib"
SCHEMA = ROOT / "models/input_schema.json"


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def forbidden(*a, **kw): pytest.fail("Ningún entrenamiento está autorizado")
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(LGBMClassifier, "fit", forbidden)


@pytest.fixture(scope="module")
def assets():
    return r.load_assets(MODEL, SCHEMA, trusted=True)


def client():
    row = dict(LIMIT_BAL=100000., SEX="1", EDUCATION="2", MARRIAGE="2", AGE=35.)
    row.update({name: 0 for name in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")})
    for i in range(1, 7): row.update({f"BILL_AMT{i}": 20000., f"PAY_AMT{i}": 2000.})
    return row


def test_python_and_asset_identity(assets):
    model, schema = assets
    assert sys.version_info[:2] == (3, 12)
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == schema["artifact"]["sha256"]
    assert r.operating_threshold(schema) == .2
    assert "Andrés" in schema["decision_provenance"]["author"]
    assert list(model.named_steps) == ["features", "prep", "model"]
    assert model.named_steps["model"].get_params()["n_estimators"] == 300


def test_single_batch_and_numeric_categories_match_pipeline(assets):
    model, schema = assets
    frame = pd.DataFrame([client(), {**client(), "PAY_0": 2}], index=["a", "b"])
    result = r.score_clients(frame, model, schema)
    normalized = r.prepare_inputs(frame, schema)
    assert all(normalized[name].dtype.kind in "fi" for name in schema["categorical"])
    np.testing.assert_allclose(result.proba_1, model.predict_proba(normalized)[:, 1])
    np.testing.assert_array_equal(result.decision, np.where(result.proba_1 >= .2, "1", "0"))
    assert result.index.tolist() == ["a", "b"] and result.proba_1.between(0, 1).all()
    assert result.attrs["operating_threshold"] == .2
    assert not any("categoría no vista" in warning for warning in result.attrs["warnings"])
    assert len(r.score_clients(frame.iloc[:1], model, schema)) == 1


def test_exact_threshold_boundary():
    probabilities = np.array([[.800001, .199999], [.8, .2], [.799999, .200001]])
    np.testing.assert_array_equal(decide_by_threshold(probabilities, ["0", "1"], .2, "1"), [0, 1, 1])


@pytest.mark.parametrize("column,value", [("SEX", "9"), ("EDUCATION", "0"), ("MARRIAGE", "x"),
    ("SEX", True), ("AGE", 200), ("AGE", "35"), ("AGE", 22.5), ("LIMIT_BAL", 0),
    ("PAY_0", 10), ("PAY_AMT1", -1), ("BILL_AMT1", np.inf)])
def test_invalid_inputs(assets, column, value):
    model, schema = assets
    with pytest.raises(InputValidationError):
        r.score_clients(pd.DataFrame([{**client(), column: value}]), model, schema)


@pytest.mark.parametrize("missing", [None, pd.NA, np.nan])
def test_missing_values_are_imputed_by_saved_pipeline(assets, missing):
    model, schema = assets
    row = {**client(), "LIMIT_BAL": missing, "EDUCATION": missing}
    assert np.isfinite(r.score_clients(pd.DataFrame([row]), model, schema).proba_1).all()


def test_unknown_source_tampering_and_extra_columns(assets, tmp_path):
    model, schema = assets
    with pytest.raises(PermissionError): r.load_assets(MODEL, SCHEMA)
    corrupted = tmp_path / "modified.joblib"
    corrupted.write_bytes(MODEL.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="SHA-256"): r.load_assets(corrupted, SCHEMA, trusted=True)
    with pytest.raises(InputValidationError): r.score_clients(pd.DataFrame([{**client(), "ID": 1}]), model, schema)


def test_streamlit_initial_load_and_submit():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    assert not app.exception and not app.error
    app.button[0].click().run()
    assert not app.exception and not app.error and len(app.metric) == 1
    assert any("20%" in item.value for item in app.markdown)
    assert any("confirmación escrita" in item.value for item in app.warning)


def test_streamlit_category_choices_and_unknown_numeric():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    app.selectbox(key="cat_EDUCATION").select("4")
    app.checkbox(key="unk_num_LIMIT_BAL").check()
    app.button[0].click().run()
    assert not app.exception and not app.error and len(app.metric) == 1


def test_headless_server_starts_and_stops():
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    process = subprocess.Popen([sys.executable, "-B", "-m", "streamlit", "run", str(ROOT / "app/streamlit_app.py"),
        "--server.headless=true", "--server.address=127.0.0.1", f"--server.port={port}",
        "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"],
        cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for _ in range(120):
            if process.poll() is not None:
                pytest.fail("Streamlit no inició: " + process.stderr.read().decode("utf-8", errors="replace"))
            try:
                with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(.25)
        else:
            pytest.fail("No respondió la salud HTTP de Streamlit local")
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=10)
