"""Ejecuta con datos reales el recorrido 07 -> 08 -> 09 de RiskVision (la lógica vive en src/proyecto/models/final_run.py).

Qué hace: valida con el entrenamiento, congela la decisión, entrena el modelo final con los 24.000 clientes, guarda un
paquete verificable fuera de Git, exporta models/model.joblib + models/input_schema.json (lo que carga la app) y,
si se pide, registra el modelo con versión y alias champion (Databricks si hay conexión en .env; si no, MLflow local).

La PRUEBA NO SE ABRE salvo que se pase --open-test-once (ya se abrió una vez; ver reports/07_umbral_y_resultados.md).
La aprobación definitiva solo se aplica con --approve-definitive y es una decisión humana de quien ejecuta.

Uso (desde la raíz):
    python scripts/run_etapas_07_09.py --approved-by "Nombre" --out resultados.json [--approve-definitive --register]
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONUTF8", "1")

from proyecto.config import PROCESSED_DIR  # noqa: E402
from proyecto.data.split import make_xy  # noqa: E402
from proyecto.models import final_run as fr  # noqa: E402
from proyecto.models import validation as v  # noqa: E402
from proyecto.runtime import diagnose_environment  # noqa: E402
from proyecto.tracking.reporting import code_revision  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-by", required=True, help="Persona o grupo responsable de autorizar el ajuste final")
    parser.add_argument("--out", required=True, help="JSON de resultados (se crea nuevo)")
    parser.add_argument("--approve-definitive", action="store_true", help="Marca el modelo como definitivo (decisión humana)")
    parser.add_argument("--register", action="store_true", help="Registra versión y alias champion (requiere --approve-definitive)")
    parser.add_argument("--open-test-once", action="store_true", help="Abre la prueba (ya se abrió una vez)")
    args = parser.parse_args()
    if args.register and not args.approve_definitive:
        sys.exit("--register exige --approve-definitive")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_dir = ROOT / "models" / "_local" / stamp  # fuera de Git (ver .gitignore)
    local_dir.mkdir(parents=True, exist_ok=False)

    diagnostic = diagnose_environment(ROOT / "requirements-serverless-06.txt", strict=False)
    versions = {**diagnostic["versions"], "python": diagnostic["python"]}
    revision = code_revision(ROOT)
    train = v.load_training(PROCESSED_DIR / "train.joblib", expected_sha256=fr.TRAIN_SHA256)
    print("Entrenamiento verificado:", train.shape, flush=True)

    stage07 = fr.validate_and_freeze(train, revision=revision, versions=versions)
    print("Configuración congelada:", stage07["settings"], flush=True)
    handoff, receipt, approved_at = fr.prepare_authorized_handoff(
        stage07["frozen"], stage07["report"], local_dir / "handoff_07_a_08.json", approved_by=args.approved_by)
    decision = fr.definitive_approval(handoff, approved_by=args.approved_by) if args.approve_definitive else handoff

    bundle, saved, inputs = fr.fit_and_save_bundle(train, decision, local_dir / "bundle", revision=revision)
    fr.export_for_app(saved["directory"], ROOT / "models")
    print("Modelo exportado para la app. Estado:", bundle.metadata["status"], flush=True)

    registration = fr.register_locally(bundle, inputs, project_root=ROOT) if args.register else None
    if registration:
        print("Registro:", registration["destino"], registration["registro"]["name"],
              "v" + registration["registro"]["version"], "alias", registration["registro"]["alias"], flush=True)

    complementary = None
    if args.open_test_once:
        complementary, test = fr.evaluate_test_once(train, PROCESSED_DIR / "test.joblib", bundle, handoff)
        print("Prueba abierta (complementaria):", flush=True)

    sample, _ = make_xy(train.sample(300, random_state=0))
    inference = fr.check_inference(ROOT / "models", sample)
    internal = stage07["internal"]
    results = {
        "created_utc": approved_at, "code": revision, "versions": versions, "train_sha256": fr.TRAIN_SHA256,
        "settings": stage07["settings"], "operating_threshold": fr.OPERATING_THRESHOLD,
        "oof_metrics": fr._table(internal["oof_metrics"]), "reserve_metrics": fr._table(internal["metrics"]),
        "calibration_recommendations": internal["recommendations"], "comparisons_reserve": internal["comparisons"],
        "importance": stage07["importance"].head(12).to_dict(orient="records"),
        "segments": stage07["segments"].astype(object).where(stage07["segments"].notna(), None).to_dict(orient="records"),
        "test_metrics": None if complementary is None else fr._table(complementary["metrics"]),
        "test_comparisons": None if complementary is None else complementary["comparisons"],
        "bundle": saved, "handoff": receipt, "registration": registration, "status": bundle.metadata["status"],
        "decision_matches_threshold": inference["decision_matches_threshold"],
        "invalid_input_rejected": inference["invalid_input_rejected"],
    }
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Resultados escritos en", args.out, flush=True)


if __name__ == "__main__":
    main()
