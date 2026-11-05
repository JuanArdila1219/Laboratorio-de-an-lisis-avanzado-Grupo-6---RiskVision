"""Pruebas sintéticas y búsquedas diminutas; nunca leen datos de RiskVision."""
from dataclasses import replace

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.trial import TrialState

from proyecto.features.build import PACKAGE_E
from proyecto.models import tuning
from proyecto.models.candidates import LOGISTIC_PARAMS
from tests.test_pipeline import _clients


@pytest.fixture
def clients():
    return _clients(80, seed=3)


@pytest.fixture
def config():
    return tuning.TuningConfig(n_trials=2, n_splits=2, n_repeats=1, seed=7)


@pytest.fixture
def space():
    return tuning.SearchSpace(n_estimators=(5, 6), learning_rate=(0.05, 0.1), num_leaves=(2, 4),
                              max_depth=(2, 3), min_child_samples=(2, 4), subsample=(0.8, 1.0),
                              colsample_bytree=(0.8, 1.0), reg_lambda=(0.0, 1.0))


def _fake_scores(factory, X, y, folds):
    model = factory().named_steps["model"]
    auc = 0.5 + model.n_estimators / 10000
    return pd.DataFrame({"roc_auc": [auc] * len(folds), "avg_precision": [0.4] * len(folds),
                         "penalty": [0.6] * len(folds)})


@pytest.mark.parametrize("name,bounds", [
    ("n_estimators", (0, 100)), ("n_estimators", (100, 10)), ("n_estimators", (2.5, 10)),
    ("learning_rate", (0, 0.1)), ("learning_rate", (0.1, np.inf)),
    ("learning_rate", (0.1, np.nan)), ("learning_rate", (0.1, 1.1)),
    ("num_leaves", (1, 8)), ("num_leaves", (9, 20)), ("max_depth", (-1, 10)),
    ("min_child_samples", (0, 20)), ("subsample", (0, 1)), ("subsample", (0.5, 1.1)),
    ("colsample_bytree", (0.5, 1.1)), ("reg_lambda", (-1, 1)),
    ("n_estimators", (True, 10)), ("n_estimators", [10, 20]),
])
def test_invalid_search_spaces_are_rejected(name, bounds):
    with pytest.raises(ValueError, match=name):
        replace(tuning.SearchSpace(), **{name: bounds})


@pytest.mark.parametrize("kwargs", [{"n_trials": 0}, {"n_splits": 1}, {"n_repeats": 0},
                                    {"seed": -1}, {"seed": 2**32}, {"model_n_jobs": 0},
                                    {"n_trials": True}, {"n_trials": 1.5}])
def test_invalid_budget_and_seed_are_rejected(kwargs):
    with pytest.raises(ValueError):
        tuning.TuningConfig(**kwargs)


def test_suggested_parameters_respect_all_bounds_and_depth_constraint():
    space = tuning.SearchSpace()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    for _ in range(20):
        params = tuning.suggest_parameters(study.ask(), space)
        assert set(params) == set(space.__dataclass_fields__)
        for name, value in params.items():
            low, high = getattr(space, name)
            assert low <= value <= high
        assert params["num_leaves"] <= 2 ** params["max_depth"]


def test_trial_pipelines_and_preprocessors_are_independent_and_keep_package_e(space):
    study = optuna.create_study()
    params = tuning.suggest_parameters(study.ask(), space)
    first = tuning.make_trial_pipeline(params, seed=12)
    second = tuning.make_trial_pipeline(params, seed=12)
    assert first is not second
    for step in ["features", "prep", "model"]:
        assert first.named_steps[step] is not second.named_steps[step]
    assert first.named_steps["features"].kw_args == {"names": PACKAGE_E, "drop": ()}
    model = first.named_steps["model"]
    assert all(model.get_params()[name] == value for name, value in params.items())
    assert model.random_state == 12 and model.n_jobs == 1
    assert model.subsample_freq == 1 and model.get_params()["deterministic"]


def test_trial_pipeline_rejects_nonsearch_parameters(space):
    params = tuning.suggest_parameters(optuna.create_study().ask(), space)
    with pytest.raises(ValueError, match="ocho"):
        tuning.make_trial_pipeline({**params, "class_weight": "balanced"})


def test_logistic_reference_is_unchanged_unfitted_and_not_optimized():
    pipe = tuning.make_reference_pipeline()
    assert pipe.named_steps["features"].kw_args["names"] == PACKAGE_E
    assert all(pipe.named_steps["model"].get_params()[name] == value for name, value in LOGISTIC_PARAMS.items())
    assert not hasattr(pipe.named_steps["model"], "classes_")
    assert "scale" in pipe.named_steps["prep"].transformers[1][1].named_steps


def test_folds_are_reproducible_stratified_disjoint_and_readonly(clients, config):
    X, y = clients
    first = tuning.make_search_folds(X, y, config)
    second = tuning.make_search_folds(X, y, config)
    for (train, valid), (again_train, again_valid) in zip(first, second):
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(valid, again_valid)
        assert set(train).isdisjoint(valid)
        assert set(train) | set(valid) == set(range(len(X)))
        assert abs(y[valid].mean() - y.mean()) <= 1 / len(valid)
        assert not train.flags.writeable and not valid.flags.writeable
    other = tuning.make_search_folds(X, y, replace(config, seed=99))
    assert not np.array_equal(first[0][1], other[0][1])


@pytest.mark.parametrize("column", ["ID", "default payment next month", "unexpected", "utilization_mean"])
def test_inputs_cannot_include_ids_target_or_precomputed_features(clients, config, column):
    X, y = clients
    with pytest.raises(ValueError, match="23 predictores"):
        tuning.make_search_folds(X.assign(**{column: 0}), y, config)


def test_invalid_labels_and_misalignment_fail_before_starting_study(clients, config, monkeypatch):
    X, y = clients
    monkeypatch.setattr(optuna, "create_study", lambda **kwargs: pytest.fail("No debe crear un estudio inválido"))
    for bad_y in [y[:-1], np.full(len(y), 2), np.full(len(y), np.nan), np.zeros(len(y))]:
        with pytest.raises(ValueError):
            tuning.run_search(X, bad_y, config=config)
    with pytest.raises(ValueError, match="índices"):
        tuning.run_search(X, pd.Series(y, index=X.index + 1), config=config)
    with pytest.raises(ValueError, match="n_splits"):
        tuning.run_search(X, np.array([1] + [0] * (len(y) - 1)), config=config)


def test_trials_use_identical_folds_and_snapshot_training_data(clients, config, space, monkeypatch):
    X, y = clients
    original = X.copy(deep=True)
    observed = []

    def inspect(factory, frame, labels, folds):
        pd.testing.assert_frame_equal(frame, original)
        observed.append(folds)
        return _fake_scores(factory, frame, labels, folds)

    monkeypatch.setattr(tuning, "score_folds", inspect)
    objective = tuning.create_objective(X, y, config=config, space=space)
    X.loc[:, "AGE"] = 999  # no puede alterar los datos capturados por el objetivo
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=1))
    study.optimize(objective, n_trials=2)
    assert observed[0] is observed[1]


def test_search_maximizes_auc_and_reproduces_proposals(clients, config, space, monkeypatch):
    X, y = clients
    monkeypatch.setattr(tuning, "score_folds", _fake_scores)
    first = tuning.run_search(X, y, config=config, space=space)
    second = tuning.run_search(X, y, config=config, space=space)
    assert first.direction == optuna.study.StudyDirection.MAXIMIZE
    assert [trial.params for trial in first.trials] == [trial.params for trial in second.trials]
    assert [trial.value for trial in first.trials] == [trial.value for trial in second.trials]
    assert first.best_value == max(trial.value for trial in first.trials)
    assert first.user_attrs["features"] == list(PACKAGE_E)
    assert first.user_attrs["metric"] == "roc_auc"
    assert all("roc_auc_per_fold" in trial.user_attrs for trial in first.trials)


def test_synthetic_search_never_loads_files_and_fits_independent_pipelines(clients, config, space, monkeypatch):
    X, y = clients
    X.loc[:9, "AGE"] = np.nan
    original = X.copy(deep=True)
    pipelines = []
    factory = tuning.make_trial_pipeline

    def record(params, **kwargs):
        pipe = factory(params, **kwargs)
        pipelines.append(pipe)
        return pipe

    def forbidden(*args, **kwargs):
        pytest.fail("La búsqueda no debe cargar ni guardar archivos de datos")

    monkeypatch.setattr(tuning, "make_trial_pipeline", record)
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(joblib, "dump", forbidden)
    monkeypatch.setattr(pd, "read_excel", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    study = tuning.run_search(X, y, config=config, space=space)
    assert len(study.trials) == config.n_trials
    assert all(trial.state == TrialState.COMPLETE for trial in study.trials)
    folds = tuning.make_search_folds(X, y, config)
    assert len(pipelines) == config.n_trials * len(folds)
    assert len({id(pipe) for pipe in pipelines}) == len(pipelines)
    assert len({id(pipe.named_steps["prep"]) for pipe in pipelines}) == len(pipelines)
    for number, pipe in enumerate(pipelines):
        train_idx, _ = folds[number % len(folds)]
        prep = pipe.named_steps["prep"]
        columns = list(prep.transformers_[1][2])
        imputer = prep.named_transformers_["num"].named_steps["impute"]
        assert imputer.statistics_[columns.index("AGE")] == pytest.approx(X.iloc[train_idx]["AGE"].median())
    pd.testing.assert_frame_equal(X, original)


def test_failed_trial_is_recorded_and_next_trial_continues(clients, config, space, monkeypatch):
    X, y = clients
    calls = []

    def sometimes_fails(factory, frame, labels, folds):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("fallo sintético de ajuste")
        return _fake_scores(factory, frame, labels, folds)

    monkeypatch.setattr(tuning, "score_folds", sometimes_fails)
    study = tuning.run_search(X, y, config=config, space=space)
    assert [trial.state for trial in study.trials] == [TrialState.FAIL, TrialState.COMPLETE]
    assert study.trials[0].value is None
    assert study.trials[0].user_attrs["failure_type"] == "RuntimeError"
    assert study.best_trial.number == 1


def test_all_failed_trials_do_not_produce_fake_best_parameters(clients, config, space, monkeypatch):
    X, y = clients

    def nonfinite(factory, frame, labels, folds):
        return pd.DataFrame({"roc_auc": [np.nan] * len(folds)})

    monkeypatch.setattr(tuning, "score_folds", nonfinite)
    study = tuning.run_search(X, y, config=config, space=space)
    assert all(trial.state == TrialState.FAIL for trial in study.trials)
    with pytest.raises(ValueError):
        _ = study.best_trial


def test_programming_errors_are_not_silenced_as_failed_trials(clients, config, space, monkeypatch):
    X, y = clients

    def programming_error(*args):
        raise KeyError("error de implementación")

    monkeypatch.setattr(tuning, "score_folds", programming_error)
    with pytest.raises(KeyError):
        tuning.run_search(X, y, config=config, space=space)
