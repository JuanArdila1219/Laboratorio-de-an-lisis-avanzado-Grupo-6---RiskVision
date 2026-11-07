import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")
from proyecto.tracking import mlflow_utils


def test_one_run_per_candidate_with_mean_metrics(tmp_path):
    csv = tmp_path / "torneo.csv"
    pd.DataFrame({"modelo": ["A", "A", "B", "B"], "pliegue": [0, 1, 0, 1], "roc_auc": [0.7, 0.8, 0.6, 0.6]}).to_csv(csv, index=False)
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    mlflow.set_experiment("prueba-torneo")
    receipts = mlflow_utils.log_tournament_runs(csv)
    assert [r["modelo"] for r in receipts] == ["A", "B"] and receipts[0]["mean_roc_auc"] == 0.75
    run = mlflow.get_run(receipts[0]["run_id"])
    assert run.data.tags["registro"] == "a_posteriori" and run.data.params["n_pliegues"] == "2"
