"""Etapa 09 con paquetes 08 sintéticos, sin servicios externos ni entrenamiento real."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from proyecto.inference import service as s
from proyecto.inference.predict import InputValidationError
from tests.test_persistence import artifact


@pytest.fixture
def session(artifact):
    *_, receipt, _bundle = artifact
    return s.open_session(receipt["directory"], trusted=True,
        manifest_sha256=receipt["manifest_sha256"], controls=s.InferenceControls(diagnostic_mode=True),
        synthetic_threshold=0.5)


def test_single_and_batch_match_saved_pipeline_and_preserve_rows(artifact, session):
    X, *_, bundle = artifact
    batch = X.iloc[[8, 2, 8]].copy()
    original = batch.copy(deep=True)
    result = s.predict_clients(batch, session)
    assert result.index.tolist() == [8, 2, 8]
    np.testing.assert_allclose(result.default_probability, bundle.model.predict_proba(batch)[:, 1])
    np.testing.assert_array_equal(result.classification, result.default_probability >= .5)
    assert result.default_probability.between(0, 1).all()
    assert result.model_status.eq("provisional").all() and result.diagnostic_only.all()
    assert result.threshold_source.str.contains("synthetic").all()
    assert result.model_version.eq(bundle.metadata["model_version"]).all()
    import pandas as pd
    pd.testing.assert_frame_equal(batch, original)
    one = s.predict_clients(batch.iloc[0].to_dict(), session)
    assert len(one) == 1 and one.default_probability.iloc[0] == result.default_probability.iloc[0]
    assert not any("categoría no vista" in warning for warning in one.attrs["warnings"])
    reordered = s.predict_clients(batch[batch.columns[::-1]], session)
    np.testing.assert_array_equal(result.default_probability, reordered.default_probability)


def test_threshold_equality_and_extreme_classifications(artifact, session):
    X, *rest = artifact
    frame = X.head(1)
    probability = session.bundle.predict_proba(frame)[0, 1]
    assert s.predict_clients(frame, replace(session, threshold=float(probability))).classification.iloc[0] == 1
    assert s.predict_clients(frame, replace(session, threshold=.000001)).classification.iloc[0] == 1
    assert s.predict_clients(frame, replace(session, threshold=.999999)).classification.iloc[0] == 0


@pytest.mark.parametrize("threshold", [None, True, "0.5", -1, 0, 1, 2, np.nan, np.inf])
def test_invalid_synthetic_threshold(artifact, threshold):
    *_, receipt, bundle = artifact
    with pytest.raises(ValueError):
        s.open_session(receipt["directory"], trusted=True, manifest_sha256=receipt["manifest_sha256"],
            controls=s.InferenceControls(diagnostic_mode=True), synthetic_threshold=threshold)


@pytest.mark.parametrize("column,value", [("SEX", 9), ("EDUCATION", 0), ("MARRIAGE", 0),
    ("PAY_0", 10), ("AGE", 200), ("AGE", 25.5), ("LIMIT_BAL", 0), ("PAY_AMT1", -1),
    ("BILL_AMT1", np.inf), ("AGE", "35"), ("SEX", "1"), ("AGE", True)])
def test_invalid_inputs_have_clear_errors(session, column, value):
    frame = s.synthetic_clients().astype(object)
    frame.loc[frame.index[0], column] = value
    with pytest.raises(InputValidationError): s.predict_clients(frame, session)


def test_missing_values_valid_absent_categories_negative_bills(session):
    frame = s.synthetic_clients()
    frame.loc[:, "LIMIT_BAL"] = np.nan
    frame["EDUCATION"] = np.nan
    result = s.predict_clients(frame, session)
    assert np.isfinite(result.default_probability).all()
    assert result.attrs["scope"] == "synthetic_diagnostic"
    assert "certeza" in result.attrs["interpretation"]


@pytest.mark.parametrize("missing", [None, pd.NA, np.nan])
def test_single_client_missing_representations(session, missing):
    row = s.synthetic_clients().iloc[0].to_dict()
    row.update(EDUCATION=missing, AGE=missing, LIMIT_BAL=missing, PAY_0=missing)
    result = s.predict_clients(row, session)
    assert np.isfinite(result.default_probability).all()


def test_valid_category_absent_during_fit_is_warning_not_rejection(tmp_path):
    from proyecto.models.candidates import make_lgbm, LGBM_PARAMS
    from proyecto.features.build import PACKAGE_E
    from proyecto.models.persistence import save_bundle
    from tests.test_pipeline import _clients
    X, y = _clients(150, seed=7)
    X["EDUCATION"] = 1
    model = make_lgbm(PACKAGE_E).fit(X, y)
    receipt = save_bundle(model, X, tmp_path / "unseen", model_name="synthetic", model_version="1",
        decision={"status": "provisional", "features": list(PACKAGE_E), "params": deepcopy(LGBM_PARAMS)}, data_scope="synthetic")
    diagnostic = s.open_session(receipt["directory"], trusted=True, manifest_sha256=receipt["manifest_sha256"],
        controls=s.InferenceControls(diagnostic_mode=True), synthetic_threshold=.5)
    result = s.predict_clients(s.synthetic_clients(), diagnostic)
    assert np.isfinite(result.default_probability).all()
    assert any("EDUCATION" in warning for warning in result.attrs["warnings"])


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "empty", "list"])
def test_bad_columns_or_container(session, case):
    frame = s.synthetic_clients()
    if case == "missing": frame = frame.drop(columns="AGE")
    if case == "extra": frame["ID"] = 1
    if case == "duplicate": frame.columns = ["AGE", *frame.columns[1:]]
    if case == "empty": frame = frame.iloc[:0]
    if case == "list": frame = frame.to_dict(orient="records")
    with pytest.raises(InputValidationError): s.predict_clients(frame, session)


def test_permissions_before_io_and_unknown_source_blocked(monkeypatch):
    monkeypatch.setattr(s, "load_bundle", lambda *a, **kw: pytest.fail("No unapproved file reads"))
    with pytest.raises(PermissionError): s.open_session("unused")
    with pytest.raises(PermissionError): s.open_session("unused", controls=s.InferenceControls(diagnostic_mode=True))
    with pytest.raises(ValueError): s.open_session("unused", trusted=True, controls=s.InferenceControls(diagnostic_mode=True))
    with pytest.raises(ValueError): s.InferenceControls(diagnostic_mode=True, allow_real_inference=True)
    with pytest.raises(ValueError): s.InferenceControls(allow_real_inference=1)


@pytest.mark.parametrize("permission", ["allow_training", "allow_test_access"])
def test_training_and_test_access_cannot_be_enabled(permission):
    with pytest.raises(PermissionError): s.InferenceControls(**{permission: True})


def test_provisional_not_usable_as_real_and_publication_blocked(artifact, session, tmp_path):
    *_, receipt, bundle = artifact
    with pytest.raises(PermissionError):
        s.open_session(receipt["directory"], trusted=True, manifest_sha256=receipt["manifest_sha256"],
            controls=s.InferenceControls(allow_real_inference=True))
    result = s.predict_clients(s.synthetic_clients(), session)
    with pytest.raises(PermissionError): s.publish_predictions(result, session, tmp_path / "predictions.json")
    assert not (tmp_path / "predictions.json").exists()
    forged = replace(session, controls=s.InferenceControls(allow_real_inference=True, allow_real_publication=True))
    with pytest.raises(PermissionError): s.predict_clients(s.synthetic_clients(), forged)
    with pytest.raises(PermissionError): s.publish_predictions(result, forged, tmp_path / "predictions.json")


def test_artifact_tampering_detected_before_joblib(artifact, monkeypatch):
    *_, receipt, bundle = artifact
    path = Path(receipt["directory"]) / "pipeline.joblib"
    path.write_bytes(path.read_bytes() + b"altered")
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: pytest.fail("Do not deserialize altered model"))
    with pytest.raises(ValueError):
        s.open_session(path.parent, trusted=True, manifest_sha256=receipt["manifest_sha256"],
                       controls=s.InferenceControls(diagnostic_mode=True), synthetic_threshold=.5)


def test_inference_in_new_python_process(artifact, session, tmp_path):
    *_, receipt, bundle = artifact
    script = """import json,sys
from proyecto.inference.service import open_session,InferenceControls,predict_clients,synthetic_clients
s=open_session(sys.argv[1],trusted=True,manifest_sha256=sys.argv[2],controls=InferenceControls(diagnostic_mode=True),synthetic_threshold=.5)
r=predict_clients(synthetic_clients(),s)
print(json.dumps({'p':r.default_probability.tolist(),'c':r.classification.tolist()}))
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    process = subprocess.run([sys.executable, "-B", "-c", script, receipt["directory"], receipt["manifest_sha256"]],
                             cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    actual = json.loads(process.stdout)
    expected = s.predict_clients(s.synthetic_clients(), session)
    np.testing.assert_allclose(actual["p"], expected.default_probability, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(actual["c"], expected.classification)


def test_approved_contract_simulation_requires_frozen_threshold(monkeypatch, session, tmp_path):
    # Aprobación simulada solo en memoria: nunca presentar este modelo sintético como real.
    metadata = deepcopy(session.bundle.metadata)
    metadata.update(status="approved_definitive", data_scope="real")
    metadata["decision"].update(status="approved_definitive", frozen_for_evaluation=True,
        definitive_approval=dict(approved=True, approved_by="simulated", approved_at="2026-11-06", validation_report_id="synthetic-simulation"),
        settings={metadata["decision"]["model"]: {"threshold": .37}})
    bundle = replace(session.bundle, metadata=metadata)
    monkeypatch.setattr(s, "load_bundle", lambda *a, **kw: bundle)
    controls = s.InferenceControls(allow_real_inference=True)
    authorized = s.open_session("simulated", trusted=True, manifest_sha256="a" * 64, controls=controls)
    assert authorized.threshold == .37
    result = s.predict_clients(s.synthetic_clients(), authorized)
    with pytest.raises(PermissionError): s.publish_predictions(result, authorized, tmp_path / "blocked.json")
    with pytest.raises(PermissionError): s.predict_clients(s.synthetic_clients(), replace(authorized, threshold=.5))
    with pytest.raises(PermissionError): s.open_session("simulated", trusted=True, manifest_sha256="a" * 64, controls=controls, synthetic_threshold=.5)
    del metadata["decision"]["settings"]
    with pytest.raises(ValueError): s.open_session("simulated", trusted=True, manifest_sha256="a" * 64, controls=controls)
