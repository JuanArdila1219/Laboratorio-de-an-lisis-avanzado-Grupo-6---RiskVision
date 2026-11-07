import pandas as pd
import pytest

from proyecto.models import summary


def _results():
    row = {c: 0.5 for c in summary.METRIC_COLUMNS}
    return {"oof_metrics": {"LightGBM original": row}, "reserve_metrics": {"LightGBM original": row},
            "test_metrics": None,
            "test_comparisons": {"Logística E": {"roc_auc": {"difference": -0.01, "low": -0.02, "high": -0.005, "valid_draws": 10}}}}


def test_metrics_table_skips_missing_sources():
    table = summary.metrics_table(_results())
    assert len(table) == 2 and set(table.columns) == set(summary.METRIC_COLUMNS)


def test_paired_difference_keeps_interval_columns():
    table = summary.paired_difference(_results())
    assert list(table.columns) == ["difference", "low", "high", "valid_draws"]


def test_demo_cases_pick_min_first_above_threshold_and_max():
    p = pd.Series([0.05, 0.19, 0.20, 0.35, 0.90], index=list("abcde"))
    cases = summary.demo_cases(p, 0.20)
    assert list(cases.values()) == ["a", "c", "e"]


def test_demo_cases_rejects_bad_threshold_or_nobody_flagged():
    p = pd.Series([0.1, 0.2])
    with pytest.raises(ValueError):
        summary.demo_cases(p, 1.5)
    with pytest.raises(ValueError):
        summary.demo_cases(p, 0.9)


def test_invalid_age_example_does_not_modify_the_original():
    row = pd.DataFrame({"AGE": [40]})
    assert summary.invalid_age_example(row)["AGE"].iloc[0] == 5 and row["AGE"].iloc[0] == 40


def test_saved_results_are_loadable_and_consistent():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    results, registry = summary.load_results(root)
    assert registry["alias"] == "champion" and results["operating_threshold"] == 0.2
    assert "Prueba (umbral 1/6)" in summary.metrics_table(results).index.get_level_values(0)
