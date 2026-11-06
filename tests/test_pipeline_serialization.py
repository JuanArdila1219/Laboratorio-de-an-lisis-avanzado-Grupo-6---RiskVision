"""Persistencia con Joblib: solo datos sintéticos y archivos temporales."""
from functools import partial
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from proyecto.config import CATEGORICAL
from proyecto.data.cleaning import PAY_COLS
from proyecto.features.build import PACKAGE_E
from proyecto.models.candidates import make_lgbm, tournament_candidates
from proyecto.models.train import build_pipeline, numeric_columns
from tests.test_pipeline import _clients


CANDIDATES = tournament_candidates()


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_every_unfitted_candidate_can_be_saved_and_loaded(name, tmp_path):
    original = CANDIDATES[name]()
    path = tmp_path / "unfitted.joblib"
    joblib.dump(original, path)
    recovered = joblib.load(path)
    assert type(recovered) is type(original)
    assert not hasattr(recovered, "classes_")
    if hasattr(recovered, "named_steps"):
        assert list(recovered.named_steps) == list(original.named_steps)
        selector = recovered.named_steps["prep"].transformers[1][2]
        assert isinstance(selector, partial) and selector.func is numeric_columns
        assert selector.keywords == original.named_steps["prep"].transformers[1][2].keywords
        assert recovered.named_steps["features"].kw_args == original.named_steps["features"].kw_args


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_fitted_candidates_preserve_predictions_after_joblib_round_trip(name, tmp_path):
    X, y = _clients(300, seed=13)
    original = CANDIDATES[name]().fit(X.iloc[:200], y[:200])
    new_clients = X.iloc[200:].copy()
    new_clients.loc[new_clients.index[0], "AGE"] = np.nan
    new_clients.loc[new_clients.index[1], "EDUCATION"] = 9
    before = original.predict_proba(new_clients)
    path = tmp_path / "fitted.joblib"
    joblib.dump(original, path)
    recovered = joblib.load(path)
    # El bosque conserva sus árboles, pero la suma paralela puede variar en
    # redondeo (~1e-16) incluso al predecir dos veces sin serialización.
    np.testing.assert_allclose(recovered.predict_proba(new_clients), before, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(recovered.predict(new_clients), original.predict(new_clients))
    np.testing.assert_array_equal(recovered.classes_, original.classes_)


@pytest.mark.parametrize("linear,pay_as_category,drop", [
    (False, False, ()),
    (True, True, ()),
    (True, True, ("SEX", "PAY_0")),
    (False, False, ("EDUCATION",)),
])
def test_round_trip_preserves_feature_order_and_learned_preprocessing(linear, pay_as_category, drop, tmp_path):
    X, y = _clients(300, seed=3)
    X["EDUCATION"] = 1
    X.loc[X.index[200:], "EDUCATION"] = 4  # categoría ausente en el ajuste
    X.loc[:9, "AGE"] = np.nan
    original = build_pipeline(DummyClassifier(strategy="prior"), features=PACKAGE_E,
                              drop=drop, linear=linear, pay_as_category=pay_as_category).fit(X.iloc[:200], y[:200])
    path = tmp_path / "preprocessing.joblib"
    joblib.dump(original, path)
    recovered = joblib.load(path)
    original_features = original.named_steps["features"].transform(X)
    pd.testing.assert_frame_equal(recovered.named_steps["features"].transform(X), original_features)
    cat_cols = [c for c in [*CATEGORICAL, *(PAY_COLS if pay_as_category else [])] if c not in drop]
    expected_numeric = [c for c in original_features.columns if c not in cat_cols]
    before, after = original.named_steps["prep"], recovered.named_steps["prep"]
    for prep in (before, after):
        assert prep.transformers_[0][2] == cat_cols
        assert prep.transformers_[1][2] == expected_numeric
        assert list(prep.named_transformers_["num"].named_steps) == (["impute", "scale"] if linear else ["impute"])
    np.testing.assert_array_equal(after.get_feature_names_out(), before.get_feature_names_out())
    # Conserva medianas, modas, categorías aprendidas y escalado sin reajustar.
    for side in ("cat", "num"):
        np.testing.assert_array_equal(after.named_transformers_[side].named_steps["impute"].statistics_,
                                      before.named_transformers_[side].named_steps["impute"].statistics_)
    for a, b in zip(after.named_transformers_["cat"].named_steps["onehot"].categories_,
                    before.named_transformers_["cat"].named_steps["onehot"].categories_):
        np.testing.assert_array_equal(a, b)
    if linear:
        for attribute in ("mean_", "scale_", "var_"):
            np.testing.assert_array_equal(getattr(after.named_transformers_["num"].named_steps["scale"], attribute),
                                          getattr(before.named_transformers_["num"].named_steps["scale"], attribute))
    def dense(matrix):
        return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    np.testing.assert_array_equal(dense(after.transform(original_features)), dense(before.transform(original_features)))


def test_saved_pipeline_loads_in_a_fresh_python_process(tmp_path):
    """La recuperación no depende de una función creada en el kernel original."""
    root = Path(__file__).resolve().parents[1]
    X, y = _clients(200, seed=8)
    pipeline = make_lgbm(PACKAGE_E).fit(X.iloc[:150], y[:150])
    model_path, input_path, output_path = (tmp_path / name for name in ("model.joblib", "input.joblib", "predictions.npy"))
    joblib.dump(pipeline, model_path)
    joblib.dump(X.iloc[150:], input_path)
    code = ("import sys,joblib,numpy as np; "
            "model=joblib.load(sys.argv[1]); X=joblib.load(sys.argv[2]); "
            "np.save(sys.argv[3],model.predict_proba(X))")
    environment = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-B", "-c", code, str(model_path), str(input_path), str(output_path)],
                   cwd=tmp_path, env=environment, check=True, capture_output=True, text=True, timeout=120)
    np.testing.assert_array_equal(np.load(output_path), pipeline.predict_proba(X.iloc[150:]))
