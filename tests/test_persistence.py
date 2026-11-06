"""Persistencia exclusivamente sintética, con temporales externos al repositorio."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from proyecto.models import persistence as p
from proyecto.models.candidates import make_lgbm, make_logistic, LGBM_PARAMS
from proyecto.features.build import PACKAGE_E
from proyecto.tracking import model_registry as r
from tests.test_pipeline import _clients


@pytest.fixture
def artifact(tmp_path):
    X, y = _clients(200, seed=42)
    model = make_lgbm(PACKAGE_E).fit(X.iloc[:150], y[:150])
    decision = dict(status="provisional", model="LightGBM original", features=list(PACKAGE_E),
                    params=deepcopy(LGBM_PARAMS), provenance="synthetic unit test")
    receipt = p.save_bundle(model, X.iloc[:150], tmp_path / "bundle", model_name="riskvision",
        model_version="synthetic-1", decision=decision, data_scope="synthetic")
    loaded = p.load_bundle(receipt["directory"], trusted=True, expected_manifest_sha256=receipt["manifest_sha256"])
    return X, y, model, decision, receipt, loaded


def test_roundtrip_full_preprocessing_and_predictions(artifact):
    X, _, model, _, receipt, loaded = artifact
    assert p.compare_predictions(model, loaded, X.iloc[150:], loaded.schema)["classes_equal"]
    assert list(loaded.model.named_steps) == ["features", "prep", "model"]
    a, b = model.named_steps["prep"], loaded.model.named_steps["prep"]
    np.testing.assert_array_equal(a.get_feature_names_out(), b.get_feature_names_out())
    np.testing.assert_array_equal(a.named_transformers_["num"].named_steps["impute"].statistics_,
                                  b.named_transformers_["num"].named_steps["impute"].statistics_)
    assert p.verify_bundle(receipt["directory"])["metadata"]["artifact_sha256"] == receipt["artifact_sha256"]
    assert loaded.metadata["validation_metrics"] is None
    assert loaded.metadata["status"] == "provisional"
    assert loaded.metadata["registered_version"] is None


@pytest.mark.parametrize("name", [*p.FILES, "manifest.json"])
def test_tampering_detected_before_deserialization(artifact, monkeypatch, name):
    *_, receipt, loaded = artifact
    path = Path(receipt["directory"]) / name
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: pytest.fail("Never unpickle modified bytes"))
    with pytest.raises(ValueError):
        p.load_bundle(path.parent, trusted=True, expected_manifest_sha256=receipt["manifest_sha256"])


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "infinite", "age", "category", "payment", "limit"])
def test_invalid_schema_rejected(artifact, change):
    X, *_, loaded = artifact
    frame = X.head().copy()
    if change == "missing": frame = frame.drop(columns="AGE")
    if change == "extra": frame["ID"] = 1
    if change == "duplicate": frame.columns = ["AGE", *frame.columns[1:]]
    if change == "infinite": frame.loc[0, "BILL_AMT1"] = np.inf
    if change == "age": frame.loc[0, "AGE"] = 20.5
    if change == "category": frame.loc[0, "EDUCATION"] = 9
    if change == "payment": frame.loc[0, "PAY_AMT1"] = -1
    if change == "limit": frame.loc[0, "LIMIT_BAL"] = 0
    with pytest.raises(ValueError): loaded.predict_proba(frame)


def test_valid_missing_unseen_category_negative_bill_and_column_order(artifact):
    X, *_, loaded = artifact
    frame = X.head().copy()
    frame.loc[0, "BILL_AMT1"] = -100
    frame.loc[1, "LIMIT_BAL"] = np.nan
    assert np.isfinite(loaded.predict_proba(frame)).all()
    np.testing.assert_allclose(loaded.predict_proba(frame), loaded.predict_proba(frame[frame.columns[::-1]]))
    schema = deepcopy(loaded.schema)
    schema["categories"]["EDUCATION"] = ["1"]
    frame["EDUCATION"] = 4
    assert p.validate_input_schema(frame, schema)


def test_untrusted_and_unapproved_loading_blocked(artifact):
    *_, receipt, loaded = artifact
    with pytest.raises(PermissionError): p.load_bundle(receipt["directory"])
    with pytest.raises(PermissionError): p.load_bundle(receipt["directory"], trusted=True, require_approved=True)
    with pytest.raises(PermissionError): p.require_approved_model(loaded.metadata)


def test_disabled_operations_do_not_touch_inputs_or_services():
    with pytest.raises(PermissionError): p.train_final_reference(None, decision=None)
    with pytest.raises(PermissionError): p.save_bundle(None, None, "unused", model_name="m", model_version="1", decision={})
    with pytest.raises(PermissionError): r.publish_bundle_artifacts(None, experiment_id="1")
    with pytest.raises(PermissionError): r.register_definitive_model(None, None, registered_name="x", experiment_id="1", source_code_dir="x")
    with pytest.raises(PermissionError): r.promote_model_version(None, {})
    with pytest.raises(ValueError): p.PersistenceControls(train_real=1)


def test_provisional_and_synthetic_cannot_be_promoted_even_with_flags(artifact):
    *_, loaded = artifact
    controls = p.PersistenceControls(True, True, True, True)
    with pytest.raises(PermissionError): r.publish_bundle_artifacts(loaded, experiment_id="1", controls=controls)
    with pytest.raises(PermissionError): r.register_definitive_model(loaded, None, registered_name="x", experiment_id="1", source_code_dir="x", controls=controls)
    with pytest.raises(PermissionError): r.promote_model_version(loaded, {}, controls=controls)


def test_no_overwrite_and_no_unfitted_model(artifact, tmp_path):
    X, _, model, decision, receipt, _ = artifact
    with pytest.raises(FileExistsError):
        p.save_bundle(model, X.iloc[:150], receipt["directory"], model_name="m", model_version="1", decision=decision, data_scope="synthetic")
    with pytest.raises(ValueError):
        p.save_bundle(make_lgbm(PACKAGE_E), X, tmp_path / "unfitted", model_name="m", model_version="1", decision=decision, data_scope="synthetic")


def test_new_python_process_recovers_and_predicts(artifact, tmp_path):
    X, _, _, _, receipt, loaded = artifact
    inputs = tmp_path / "inputs.json"
    inputs.write_text(X.head().to_json(orient="split"), encoding="utf-8")
    script = """import json,sys,pandas as pd
from io import StringIO
from pathlib import Path
from proyecto.models.persistence import load_bundle
b=load_bundle(sys.argv[1],trusted=True,expected_manifest_sha256=sys.argv[2])
x=pd.read_json(StringIO(Path(sys.argv[3]).read_text()),orient='split')
print(json.dumps({'classes':b.predict(x).tolist(),'probabilities':b.predict_proba(x).tolist()}))
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-B", "-c", script, receipt["directory"], receipt["manifest_sha256"], str(inputs)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    np.testing.assert_array_equal(actual["classes"], loaded.predict(X.head()))
    np.testing.assert_allclose(actual["probabilities"], loaded.predict_proba(X.head()), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("factory", [make_logistic, make_lgbm])
def test_existing_reference_pipelines_and_calibration(factory, tmp_path):
    X, y = _clients(160)
    model = factory(PACKAGE_E)
    model = CalibratedClassifierCV(model, method="sigmoid", cv=2, ensemble=False).fit(X, y)
    base = model.calibrated_classifiers_[0].estimator.named_steps["model"]
    decision = dict(status="provisional", features=list(PACKAGE_E), params=base.get_params(), provenance="synthetic")
    receipt = p.save_bundle(model, X, tmp_path / "calibrated", model_name="m", model_version="1", decision=decision, data_scope="synthetic")
    recovered = p.load_bundle(receipt["directory"], trusted=True)
    p.compare_predictions(model, recovered, X.head(), recovered.schema)


def test_versions_checked_before_unpickle(artifact, monkeypatch):
    *_, receipt, loaded = artifact
    versions = p.dependency_versions()
    versions["joblib"] = "incompatible"
    monkeypatch.setattr(p, "dependency_versions", lambda: versions)
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: pytest.fail("incompatible load"))
    with pytest.raises(RuntimeError): p.load_bundle(receipt["directory"], trusted=True)


def test_synthetic_demonstration(tmp_path):
    assert p.synthetic_persistence_check(tmp_path / "demo")["check"]["probabilities_equivalent"]


def test_final_fit_reuses_frozen_protocol_without_fitting(monkeypatch):
    X, _ = _clients(20)
    X.attrs["source_sha256"] = "a" * 64
    decision = dict(model=p.ORIGINAL, frozen_for_evaluation=True, params=deepcopy(LGBM_PARAMS),
        features=list(PACKAGE_E), training_sha256="a" * 64,
        settings={p.ORIGINAL: {"calibration": "sigmoid"}})
    calls = []
    def fake_fit(frame, factory, **kw):
        calls.append(kw)
        assert factory().named_steps["model"].get_params()["n_estimators"] == LGBM_PARAMS["n_estimators"]
        return "not fitted; simulation"
    monkeypatch.setattr(p, "fit_training_candidate", fake_fit)
    assert p.train_final_reference(X, decision=decision, controls=p.PersistenceControls(train_real=True)) == "not fitted; simulation"
    assert calls[0]["calibrated"] is True
    decision["training_sha256"] = "b" * 64
    with pytest.raises(ValueError): p.train_final_reference(X, decision=decision, controls=p.PersistenceControls(train_real=True))
