"""Registra en MLflow una ejecución por candidato del torneo (etapa 05) con las métricas por pliegue ya medidas.

Registro a posteriori: no recalcula nada. Destino: Databricks si `.env` tiene conexión; si no, MLflow local.
Uso:  python scripts/log_torneo_mlflow.py --out recibo.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONUTF8", "1")

from proyecto.tracking.mlflow_utils import configure_mlflow, log_tournament_runs, set_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--experiment", default="riskvision-05-torneo")
    args = parser.parse_args()
    where = configure_mlflow()
    experiment = set_experiment(args.experiment)
    receipts = log_tournament_runs(ROOT / "reports/05_torneo_por_pliegue.csv")
    Path(args.out).write_text(json.dumps({"destino": where, "experimento": experiment, "registro": "a_posteriori",
                                          "fuente": "reports/05_torneo_por_pliegue.csv", "candidatos": receipts},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print("Destino:", where, "| ejecuciones registradas:", len(receipts))


if __name__ == "__main__":
    main()
