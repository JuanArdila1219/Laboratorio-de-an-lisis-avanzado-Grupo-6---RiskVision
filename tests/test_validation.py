"""Etapa 07 con clientes sintéticos; nunca abre datos del proyecto ni ejecuta Optuna."""
from dataclasses import asdict
from functools import partial
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from proyecto.config import ID_COL, TARGET
from proyecto.config import CATEGORICAL
from proyecto.features.build import PACKAGE_E
from proyecto.models.candidates import LGBM_PARAMS, make_lgbm, make_logistic
from proyecto.models.train import build_pipeline
from proyecto.models import validation as v
from tests.test_pipeline import _clients


@pytest.fixture
def frame():
    X, _ = _clients(120, seed=15)
    X.index = np.arange(2000, 2120)
    X[ID_COL] = np.arange(1000, 1120)
    X[TARGET] = (X["PAY_0"] >= 1).astype(int)
    X.attrs.update(role="synthetic_training", source_sha256="synthetic")
    return X


def decision():
    return v.provisional_decision(revision={"branch": "synthetic", "commit": "abc"},
                                  versions={"scope": "synthetic"}, training_sha256="synthetic")


def test_known_metrics_and_cost_are_correct():
    y, p = [0, 0, 1, 1], [.1, .8, .4, .9]
    result = v.metrics(y, p, .5)
    assert result["roc_auc"] == .75
    assert result["avg_precision"] == pytest.approx(5 / 6)
    assert result["precision"] == result["recall"] == result["f1"] == .5
    assert result["brier"] == pytest.approx(.255)
    assert result["log_loss"] == pytest.approx(-(2 * np.log(.9) + np.log(.2) + np.log(.4)) / 4)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["penalty"] == 1.5 and result["flagged_share"] == .5
    assert result["recall_top"] == .5


@pytest.mark.parametrize("probability", [[np.nan, .1], [-.1, .5], [1.1, .5], [[.1], [.2]]])
def test_invalid_probabilities_are_rejected(probability):
    with pytest.raises(ValueError):
        v.metrics([0, 1], probability)


@pytest.mark.parametrize("threshold", [None, -.1, 1.1, np.inf, True, "0.5"])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        v.metrics([0, 1], [.1, .9], threshold)


def test_single_class_has_no_invented_auc():
    result = v.metrics([0, 0], [.1, .8], .5)
    assert result["single_class"] and result["roc_auc"] is None and result["avg_precision"] is None
    assert result["confusion_matrix"] == [[1, 1], [0, 0]]


def test_auc_bootstrap_is_paired_reproducible_and_recalculates_auc():
    y, ref, variant, ids = [0, 0, 1, 1], [.9, .8, .2, .1], [.1, .2, .8, .9], [11, 12, 13, 14]
    result = v.paired_bootstrap(y, ref, variant, ids, n_boot=80)
    assert result == v.paired_bootstrap(y, ref, variant, ids, n_boot=80)
    assert result["difference"] == result["low"] == result["high"] == 1
    assert 2 <= result["valid_draws"] < 80
    reverse = v.paired_bootstrap(y, variant, ref, ids, n_boot=80)
    assert reverse["difference"] == -1 and reverse["low"] == -result["high"]
    identical = v.paired_bootstrap(y, ref, ref, ids, n_boot=80)
    assert identical["difference"] == identical["low"] == identical["high"] == 0


def test_penalty_bootstrap_has_its_own_sign_and_all_draws():
    result = v.paired_bootstrap([0, 0, 1, 1], [.9, .8, .2, .1], [.1, .2, .8, .9],
                              [1, 2, 3, 4], metric="penalty", reference_threshold=.5,
                              variant_threshold=.5, n_boot=80)
    assert result["difference"] == -3 and result["valid_draws"] == 80
    assert result["metric"] == "penalty" and result["high"] < 0


def test_bootstrap_rejects_duplicate_clients_and_single_class_auc():
    with pytest.raises(AssertionError):
        v.paired_bootstrap([0, 1], [.1, .9], [.2, .8], [1, 1], n_boot=20)
    with pytest.raises(ValueError):
        v.paired_bootstrap([0, 0], [.1, .9], [.2, .8], [1, 2], n_boot=20)


def test_threshold_selection_uses_training_oof_only_and_deterministic_ties():
    oof = pd.DataFrame({ID_COL: [1, 2, 3, 4], "y_true": [0, 0, 1, 1], "proba": [.1, .8, .4, .9]})
    oof.attrs["role"] = "training_oof"
    result = v.select_threshold(oof, [.2, .3, .5])
    assert result["threshold"] == .3
    assert result["status"] == "recommendation_pending_review"
    table = v.threshold_analysis(oof.y_true, oof.proba, [.2, .3, .5])
    assert table["roc_auc"].nunique() == 1
    for role in ["internal_reserve", "exposed_test", None]:
        oof.attrs["role"] = role
        with pytest.raises(PermissionError):
            v.select_threshold(oof)


def test_calibration_recommendation_does_not_promote_model():
    raw = pd.DataFrame({ID_COL: [1, 2, 3, 4], "y_true": [0, 0, 1, 1], "proba": [.01, .99, .99, .01]})
    raw.attrs["role"] = "training_oof"
    calibrated = raw.copy()
    calibrated["proba"] = [.4, .4, .6, .6]
    recommendation = v.recommend_calibration(raw, calibrated)
    assert recommendation["recommend_sigmoid"] and recommendation["status"].endswith("pending_review")
    calibrated.attrs["role"] = "exposed_test"
    with pytest.raises(PermissionError):
        v.recommend_calibration(raw, calibrated)


def test_default_permissions_block_before_data_or_factory_access(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No acceder a datos ni construir modelos sin permiso")
    monkeypatch.setattr(joblib, "load", forbidden)
    controls = v.ExecutionControls()
    with pytest.raises(PermissionError):
        v.fit_training_candidate(None, forbidden, controls=controls)
    with pytest.raises(PermissionError):
        v.crossfit_probabilities(None, forbidden, controls=controls)
    with pytest.raises(PermissionError):
        v.evaluate_internal(None, controls=controls)
    with pytest.raises(PermissionError):
        v.fit_training_candidate(None, forbidden, controls=v.ExecutionControls(fit_real=True), calibrated=True)
    with pytest.raises(PermissionError):
        v.load_complementary_test(tmp_path / "never-read", controls=controls, decision=None, training_ids=[])
    with pytest.raises(PermissionError):
        v.evaluate_complementary({}, None, controls=controls, decision=None)
    with pytest.raises(PermissionError):
        v.export_definitive_model(None, tmp_path / "blocked" / "model.joblib", controls=controls, decision=None)
    assert not (tmp_path / "blocked").exists()


def test_oof_imputation_learns_only_fold_training_and_clients_are_disjoint(frame, monkeypatch):
    frame.loc[frame.index[:12], "AGE"] = np.nan
    fits = []
    real_fit = Pipeline.fit
    def audited_fit(self, X, y=None, **kwargs):
        result = real_fit(self, X, y, **kwargs)
        prep = self.named_steps["prep"]
        cols = list(prep.transformers_[1][2])
        median = prep.named_transformers_["num"].named_steps["impute"].statistics_[cols.index("AGE")]
        fits.append((self, list(X.index), median))
        return result
    monkeypatch.setattr(Pipeline, "fit", audited_fit)
    oof = v.crossfit_probabilities(frame, partial(make_logistic, PACKAGE_E),
                                   controls=v.ExecutionControls(fit_real=True), n_splits=3)
    assert len(fits) == 3 and len({id(item[0]) for item in fits}) == 3
    assert oof[ID_COL].tolist() == frame[ID_COL].tolist() and oof.proba.notna().all()
    for (_, indices, median), audit in zip(fits, oof.attrs["folds"]):
        assert median == pytest.approx(frame.loc[indices, "AGE"].median())
        assert set(audit["train_ids"]).isdisjoint(audit["validation_ids"])
        assert set(frame.loc[indices, ID_COL]) == set(audit["train_ids"])
    frame.attrs["role"] = "internal_reserve"
    with pytest.raises(PermissionError):
        v.fit_training_candidate(frame, partial(make_logistic, PACKAGE_E), controls=v.ExecutionControls(fit_real=True))


def test_calibrator_training_never_sees_outer_validation(frame, monkeypatch):
    fits = []
    real_fit = Pipeline.fit
    def capture(self, X, y=None, **kwargs):
        fits.append(set(frame.loc[X.index, ID_COL]))
        return real_fit(self, X, y, **kwargs)
    monkeypatch.setattr(Pipeline, "fit", capture)
    oof = v.crossfit_probabilities(frame, partial(make_logistic, PACKAGE_E),
        controls=v.ExecutionControls(fit_real=True, calibrate_real=True), calibrated=True, n_splits=3)
    assert len(fits) == 12  # tres ajustes para calibración OOF + ajuste de base por pliegue externo
    for number, audit in enumerate(oof.attrs["folds"]):
        for fitted_ids in fits[number * 4:(number + 1) * 4]:
            assert fitted_ids <= set(audit["train_ids"])
            assert fitted_ids.isdisjoint(audit["validation_ids"])
    assert oof.proba.between(0, 1).all()


def test_segments_are_descriptive_and_do_not_drop_clients(frame):
    p = np.linspace(.1, .9, len(frame))
    result = v.segment_metrics(frame, p)
    for name in ["SEX", "AGE"]:
        assert result.loc[result.variable == name, "n"].sum() == len(frame)


def test_permutation_is_reproducible_and_does_not_refit(frame, monkeypatch):
    controls = v.ExecutionControls(fit_real=True)
    model = v.fit_training_candidate(frame.iloc[:90], partial(make_logistic, PACKAGE_E), controls=controls)
    monkeypatch.setattr(Pipeline, "fit", lambda *a, **kw: pytest.fail("Interpretar no ajusta"))
    first = v.interpretation(model, frame.iloc[90:], controls=controls, n_repeats=2)
    second = v.interpretation(model, frame.iloc[90:], controls=controls, n_repeats=2)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 23 and ID_COL not in set(first.feature)
    assert not set(PACKAGE_E) & set(first.feature)


def test_calibration_curve_and_plot_are_prepared():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curve = v.calibration_diagnostics([0, 0, 1, 1], [.1, .3, .7, .9], n_bins=4)
    assert curve["curve"].n_clients.sum() == 4
    figure = v.plot_calibration({"synthetic": curve})
    assert len(figure.axes[0].lines) == 2
    plt.close(figure)


def test_training_loader_checks_hash_without_recreating_data(frame, tmp_path, monkeypatch):
    path = tmp_path / "train.joblib"
    joblib.dump(frame, path)
    before = path.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    recovered = v.load_training(path, expected_sha256=digest, expected_rows=len(frame))
    pd.testing.assert_frame_equal(recovered, frame, check_flags=False)
    assert path.read_bytes() == before
    monkeypatch.setattr(joblib, "load", lambda *a: pytest.fail("No deserializar huella inválida"))
    with pytest.raises(ValueError, match="huella"):
        v.load_training(path, expected_sha256="incorrect")


def test_freeze_and_export_do_not_turn_provisional_into_definitive(frame, tmp_path):
    settings = {name: {"calibration": "none", "threshold": .5} for name in (v.ORIGINAL, v.REFERENCE)}
    with pytest.raises(PermissionError):
        v.freeze_for_evaluation(decision(), settings)
    frozen = v.freeze_for_evaluation(decision(), settings, approved=True)
    settings[v.ORIGINAL]["threshold"] = .9
    assert frozen["settings"][v.ORIGINAL]["threshold"] == .5 and frozen["status"] == "provisional"
    with pytest.raises(PermissionError):
        v.export_definitive_model(None, tmp_path / "blocked.joblib",
            controls=v.ExecutionControls(fit_real=True, export_definitive=True), decision=frozen)
    assert not (tmp_path / "blocked.joblib").exists()


def test_optimized_evidence_needs_complete_approved_own_protocol():
    factories = v.candidate_factories()
    assert set(factories) == {v.ORIGINAL, v.REFERENCE}
    original = factories[v.ORIGINAL]().named_steps["model"].get_params()
    assert all(original[k] == value for k, value in LGBM_PARAMS.items())
    with pytest.raises(PermissionError):
        v.candidate_factories(optimized_evidence={})
    with pytest.raises(ValueError):
        v.candidate_factories(optimized_evidence={"finalista": "LightGBM", "origen": "partida"},
                              approve_optimized=True, training_sha256="synthetic")
    from proyecto.models.tuning import SearchSpace, TuningConfig
    params = dict(n_estimators=100, learning_rate=.03, num_leaves=8, max_depth=3,
                  min_child_samples=20, subsample=.8, colsample_bytree=.8, reg_lambda=.1)
    evidence = {"metric": "roc_auc", "direction": "MAXIMIZE", "features": list(PACKAGE_E),
                "training_sha256": "synthetic", "config": asdict(TuningConfig()),
                "search_space": asdict(SearchSpace()), "reserve": {"seed": 7, "share": .25},
                "code": {"branch": "revision-h2-riskvision", "commit": "abc"},
                "report_id": "synthetic", "best_trial": 0,
                "trials": [{"number": 0, "state": "COMPLETE", "params": params, "roc_auc_cv": .7}],
                "best_cv_auc": .7, "best_params_candidate": params}
    assert v.OPTIMIZED in v.candidate_factories(optimized_evidence=evidence, approve_optimized=True,
                                               training_sha256="synthetic")
    evidence["trials"] = []
    with pytest.raises(ValueError, match="trazabilidad"):
        v.candidate_factories(optimized_evidence=evidence, approve_optimized=True, training_sha256="synthetic")


def test_end_to_end_internal_synthetic_report_and_pipeline_round_trip(frame, monkeypatch, tmp_path):
    factories = {v.ORIGINAL: partial(make_logistic, PACKAGE_E),
                 v.REFERENCE: partial(build_pipeline, DummyClassifier(strategy="prior"), features=PACKAGE_E)}
    monkeypatch.setattr(v, "candidate_factories", lambda **kw: factories)
    controls = v.ExecutionControls(fit_real=True, calibrate_real=True)
    result = v.evaluate_internal(frame, controls=controls, n_splits=3, n_boot=30)
    assert len(result["development"]) == 90 and len(result["reserve"]) == 30
    assert set(result["development"][ID_COL]).isdisjoint(result["reserve"][ID_COL])
    assert len(result["metrics"]) == 4 and len(result["recommendations"]) == 2
    report = v.validation_report(decision(), controls=controls, internal=result)
    assert report["status"] == "pending_review" and not report["definitive"]
    assert report["complementary_evaluation"] is None
    json.dumps(report, allow_nan=False)
    path = tmp_path / "synthetic-pipeline.joblib"
    joblib.dump(result["models"][v.ORIGINAL], path)
    X, _ = v.make_xy(result["reserve"])
    np.testing.assert_allclose(joblib.load(path).predict_proba(X), result["models"][v.ORIGINAL].predict_proba(X))


def test_complementary_metrics_with_synthetic_frame_never_select_threshold(frame):
    fit_controls = v.ExecutionControls(fit_real=True)
    model = v.fit_training_candidate(frame.iloc[:90], partial(make_logistic, PACKAGE_E), controls=fit_controls)
    frozen = v.freeze_for_evaluation(decision(), {name: {"calibration": "none", "threshold": .5}
        for name in (v.ORIGINAL, v.REFERENCE)}, approved=True)
    result = v.evaluate_complementary({v.ORIGINAL: model, v.REFERENCE: model}, frame.iloc[90:],
        controls=v.ExecutionControls(evaluate_test=True), decision=frozen, n_boot=30)
    assert not result["independent_evaluation"]
    assert result["comparisons"][v.REFERENCE]["roc_auc"]["difference"] == 0


def test_test_reader_requires_frozen_decision_before_any_file_access(tmp_path, monkeypatch):
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: pytest.fail("No leer sin decisiones"))
    monkeypatch.setattr(v.Path, "read_bytes", lambda *a: pytest.fail("No leer ni la huella"))
    with pytest.raises(PermissionError, match="congeladas"):
        v.load_complementary_test(tmp_path / "blocked", controls=v.ExecutionControls(evaluate_test=True),
                                  decision=decision(), training_ids=[])


def test_test_reader_contract_with_bytes_in_memory_only(frame, tmp_path, monkeypatch):
    """No se crea/lee ningún test.joblib: se simulan los bytes y joblib.load."""
    frozen = v.freeze_for_evaluation(decision(), {name: {"calibration": "none", "threshold": .5}
        for name in (v.ORIGINAL, v.REFERENCE)}, approved=True)
    monkeypatch.setattr(v.Path, "read_bytes", lambda *a: b"synthetic-bytes-only")
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: frame.copy())
    controls = v.ExecutionControls(evaluate_test=True)
    recovered = v.load_complementary_test(tmp_path / "test.joblib", controls=controls, decision=frozen,
                                          training_ids=[9999], expected_rows=len(frame))
    assert recovered.attrs["role"] == "exposed_test"
    with pytest.raises(ValueError, match="compartidos"):
        v.load_complementary_test(tmp_path / "test.joblib", controls=controls, decision=frozen,
                                  training_ids=frame[ID_COL], expected_rows=len(frame))
    with pytest.raises(PermissionError):
        v.fit_training_candidate(recovered, partial(make_logistic, PACKAGE_E), controls=v.ExecutionControls(fit_real=True))


def test_explicitly_approved_synthetic_export_is_exclusive(frame, tmp_path):
    controls = v.ExecutionControls(fit_real=True, export_definitive=True)
    model = v.fit_training_candidate(frame, partial(make_logistic, PACKAGE_E), controls=controls)
    frozen = v.freeze_for_evaluation(decision(), {name: {"calibration": "none", "threshold": .5}
        for name in (v.ORIGINAL, v.REFERENCE)}, approved=True)
    frozen["status"] = "approved_definitive"  # autorización sintética, no decisión real del proyecto
    path = tmp_path / "synthetic-approved.joblib"
    v.export_definitive_model(model, path, controls=controls, decision=frozen)
    X, _ = v.make_xy(frame)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(X), model.predict_proba(X))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        v.export_definitive_model(model, path, controls=controls, decision=frozen)
    assert path.read_bytes() == before


def test_lightgbm_pipeline_accepts_processed_categorical_training_schema(frame):
    for column in CATEGORICAL:
        frame[column] = frame[column].astype("category")
    result = v.crossfit_probabilities(frame, partial(make_lgbm, PACKAGE_E),
        controls=v.ExecutionControls(fit_real=True), n_splits=3)
    assert result.proba.between(0, 1).all()
    assert result[ID_COL].tolist() == frame[ID_COL].tolist()
