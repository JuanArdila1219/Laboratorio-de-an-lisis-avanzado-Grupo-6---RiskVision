"""Integración 07→08→09: solo clientes, métricas y autorizaciones SINTÉTICOS."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from proyecto.config import ID_COL, TARGET
from proyecto.models import decision as d, persistence as p, validation as v
from proyecto.inference import service as s
from tests.test_pipeline import _clients


def synthetic_handoff(calibration="none", training_sha256="a" * 64):
    decision = v.provisional_decision(revision={"branch": "synthetic-test", "commit": "synthetic-test"},
        versions=p.dependency_versions(), training_sha256=training_sha256)
    settings = {name: {"calibration": calibration if name == v.ORIGINAL else "none", "threshold": .4}
                for name in (v.ORIGINAL, v.REFERENCE)}
    frozen = v.freeze_for_evaluation(decision, settings, approved=True)
    report = v.validation_report(frozen, controls=v.ExecutionControls())
    label = v.ORIGINAL + (" + sigmoid" if calibration == "sigmoid" else "")
    # Métricas de vectores artificiales, no cifras reales ni de Estefanía.
    synthetic_metrics = v.metrics([0, 1, 0, 1], [.2, .7, .3, .8], threshold=.4)
    report["internal_evaluation"] = {"scope": "SYNTHETIC_TEST_ONLY", "metrics": {label: synthetic_metrics}}
    return d.prepare_handoff(frozen, report, authorize_final_fit=True,
        approved_by="SYNTHETIC_REVIEWER", approved_at="2026-11-06T00:00:00Z")


def approve_synthetic(handoff):
    return d.approve_for_inference(handoff, approved=True, approval={"approved": True,
        "approved_by": "SYNTHETIC_REVIEWER", "approved_at": "2026-11-06T00:00:00Z",
        "validation_report_id": handoff["handoff"]["validation_report"]["report_id"],
        "configuration_sha256": handoff["handoff"]["configuration_sha256"]})


def test_adoption_is_credited_and_never_own_optimization():
    result = d.adopted_experiment()
    assert result["author"] == "Estefanía" and result["commit"] == d.ESTEFANIA_COMMIT
    assert result["params"] == d.ADOPTED_PARAMS == p.LGBM_PARAMS
    assert result["own_real_optimization_executed"] is False
    assert result["reported_metrics"]["best_cv_auc"] == .7869
    result["params"]["n_estimators"] = 1
    assert d.adopted_experiment()["params"]["n_estimators"] == 300


def test_freeze_handoff_and_definitive_approval_are_distinct(tmp_path):
    handoff = synthetic_handoff()
    assert handoff["status"] == "provisional" and handoff["definitive"] is False
    with pytest.raises(PermissionError): d.prepare_handoff(handoff, {})
    with pytest.raises(PermissionError): d.approve_for_inference(handoff)
    with pytest.raises(PermissionError): d.approve_for_inference(handoff, approved=True)
    approved = approve_synthetic(handoff)
    assert approved["status"] == "approved_definitive"
    assert handoff["status"] == "provisional"  # nunca muta el original
    receipt = d.write_handoff(handoff, tmp_path / "handoff.json")
    loaded = d.read_handoff(receipt["path"], trusted=True, expected_sha256=receipt["sha256"])
    assert loaded == handoff
    with pytest.raises(FileExistsError): d.write_handoff(handoff, receipt["path"])
    with pytest.raises(PermissionError): d.read_handoff(receipt["path"], expected_sha256=receipt["sha256"])
    Path(receipt["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError): d.read_handoff(receipt["path"], trusted=True, expected_sha256=receipt["sha256"])


@pytest.mark.parametrize("case", ["params", "features", "threshold", "calibration", "hash", "versions", "metrics", "author", "utc", "owner", "own_execution"])
def test_incompatible_contracts_fail(case):
    decision = synthetic_handoff()
    if case == "params": decision["params"]["num_leaves"] = 31
    if case == "features": decision["features"] = []
    if case == "threshold": decision["settings"][v.ORIGINAL]["threshold"] = 1
    if case == "calibration": decision["settings"][v.ORIGINAL]["calibration"] = "isotonic"
    if case == "hash": decision["training_sha256"] = "b" * 64
    if case == "versions": decision["versions"]["joblib"] = "changed"
    if case == "metrics": decision["handoff"]["validation_report"]["internal_evaluation"] = None
    if case == "author": decision["handoff"]["final_fit_authorization"]["approved_by"] = ""
    if case == "utc": decision["handoff"]["final_fit_authorization"]["approved_at"] = "2026-11-06"
    if case == "owner": decision["handoff"]["final_fit_owner"] = "07"
    if case == "own_execution": decision["own_real_optimization_executed"] = True
    with pytest.raises((ValueError, PermissionError)): d.validate_handoff(decision)


def test_final_fit_requires_handoff_and_environment_before_fitting(monkeypatch):
    frame, _ = _clients(20)
    frame.attrs["source_sha256"] = "a" * 64
    with pytest.raises(ValueError): p.train_final_reference(frame, decision={}, controls=p.PersistenceControls(train_real=True))
    with pytest.raises(ValueError): p.train_final_reference(frame, decision=synthetic_handoff(), controls=p.PersistenceControls(train_real=True))
    with pytest.raises(PermissionError): p.train_final_reference(None, decision=synthetic_handoff())


@pytest.mark.parametrize("calibration", ["none", "sigmoid"])
def test_synthetic_final_fit_bundle_and_inference_end_to_end(tmp_path, calibration):
    # 24.000 clientes artificiales; se ejercita la misma exigencia dimensional de producción.
    frame, y = _clients(24000, seed=24)
    frame[ID_COL] = np.arange(len(frame))
    frame[TARGET] = y
    frame = frame.loc[:, [ID_COL, *p.PREDICTORS, TARGET]]
    training_path = tmp_path / "train.joblib"  # dataset enteramente artificial
    joblib.dump(frame, training_path)
    training_sha256 = hashlib.sha256(training_path.read_bytes()).hexdigest()
    frame = v.load_training(training_path, expected_sha256=training_sha256)
    handoff = synthetic_handoff(calibration, training_sha256=training_sha256)
    approved = approve_synthetic(handoff)  # aprobación ficticia solo para simular el contrato real
    model = p.train_final_reference(frame, decision=approved, controls=p.PersistenceControls(train_real=True))
    X = frame.drop(columns=[ID_COL, TARGET])
    receipt = p.save_bundle(model, X, tmp_path / "approved-simulation", model_name="synthetic_only", model_version="1",
        decision=approved, controls=p.PersistenceControls(publish_real_artifacts=True), data_scope="real",
        validation_metrics=d.selected_validation_metrics(approved), revision=approved["code"])
    session = s.open_session(receipt["directory"], trusted=True, manifest_sha256=receipt["manifest_sha256"],
        controls=s.InferenceControls(allow_real_inference=True))
    result = s.predict_clients(X.head(3), session)
    np.testing.assert_allclose(result.default_probability, model.predict_proba(X.head(3))[:, 1])
    assert result.threshold.eq(.4).all()
    assert session.bundle.metadata["decision"]["own_real_optimization_executed"] is False
    assert all(key.startswith("internal_reserve_") for key in session.bundle.metadata["validation_metrics"])
    with pytest.raises(PermissionError): s.publish_predictions(result, session, tmp_path / "forbidden.json")


def test_forging_only_definitive_status_is_not_approval():
    handoff = synthetic_handoff()
    handoff["status"] = "approved_definitive"
    with pytest.raises(PermissionError): d.validate_handoff(handoff)


def test_calibration_mismatch_rejected_before_saving(tmp_path):
    X, y = _clients(120)
    model = p.make_lgbm(p.PACKAGE_E).fit(X, y)
    with pytest.raises(ValueError, match="calibración"):
        p.save_bundle(model, X, tmp_path / "mismatch", model_name="m", model_version="1",
            decision=synthetic_handoff("sigmoid"), data_scope="synthetic")


def test_07_freezing_cell_has_no_final_fit_path(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks/07_validacion_interpretacion.ipynb").read_text(encoding="utf-8"))
    source = "".join(nb["cells"][9]["source"])
    assert "fit_training_candidate" not in source and ".fit(" not in source
    assert "AUTHORIZE_FINAL_FIT_HANDOFF = False" in source
    def forbidden(*a, **kw): pytest.fail("07 no debe ejecutar el ajuste final al congelar")
    monkeypatch.setattr(v, "fit_training_candidate", forbidden)
    frozen = synthetic_handoff()
    source = source.replace("APPROVE_CONFIGURATION_FOR_EVALUATION = False", "APPROVE_CONFIGURATION_FOR_EVALUATION = True")
    source = source.replace("FROZEN_SETTINGS = None", "FROZEN_SETTINGS = " + repr(frozen["settings"]))
    ns = {"validation": v, "DECISION_PROVISIONAL": frozen,
          "FACTORIES": {v.ORIGINAL: None, v.REFERENCE: None}}
    exec(compile(source, "07-freeze-only", "exec"), ns)
    assert ns["FROZEN_DECISION"]["status"] == "provisional"
    assert ns["FITTED_FOR_EVALUATION"] == {}
