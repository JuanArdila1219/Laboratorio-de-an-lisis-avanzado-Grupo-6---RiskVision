"""Recorrido real 07 -> 08 -> 09 de RiskVision, dividido en pasos que un notebook o un script puede orquestar.

Cada paso hace una sola cosa y no toma decisiones por su cuenta:
  validate_and_freeze      07  validación interna (solo entrenamiento) y congelación de la decisión
  prepare_authorized_handoff 07->08  contrato firmado (responsable y fecha UTC)
  fit_and_save_bundle      08  ajuste final con los 24.000 clientes y paquete verificable
  build_app_schema/export_for_app  copia lo que carga la app (models/model.joblib + input_schema.json)
  register_locally         08  registro en MLflow (versión + alias) con el respaldo local del curso
  evaluate_test_once       07b abre la prueba UNA sola vez (ya expuesta: no es independiente)
  check_inference          09  comprobaciones de la inferencia sobre el modelo exportado

Nada de aquí sube datos a internet: el registro usa el destino que devuelva `configure_mlflow`
(Databricks si hay conexión en `.env`; si no, MLflow local).
"""
import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from proyecto.config import ID_COL
from proyecto.data.split import make_xy
from proyecto.models import validation as v
from proyecto.models.decision import (approve_for_inference, prepare_handoff, read_handoff,
                                      selected_validation_metrics, write_handoff)
from proyecto.models.persistence import (PersistenceControls, compare_predictions, load_bundle,
                                        save_bundle, train_final_reference)

TRAIN_SHA256 = "4e71503c0072bcbf3ec76cf862614c2e322a79a08e21db02f3225b9398028568"
MODEL_NAME = "riskvision_lightgbm_e"

# Punto de trabajo de la app (ver reports/07_umbral_y_resultados.md): el umbral más alto de la grilla cuya
# penalización 5:1 en validación cruzada queda a menos del 2 % del mínimo. NO sale de la prueba.
OPERATING_THRESHOLD = 0.20
OPERATING_RULE = ("marca incumple (1) si la probabilidad es >= decision_threshold; punto de trabajo del grupo "
                  "elegido con validacion cruzada, no un costo medido del banco")


APP_TITLE = "RiskVision · Riesgo de impago de tarjeta de crédito"
APP_LABELS = {
    "LIMIT_BAL": "Límite de crédito (NT$)", "SEX": "Sexo (1 = hombre, 2 = mujer)",
    "EDUCATION": "Educación (1 posgrado, 2 universidad, 3 secundaria, 4 otro)",
    "MARRIAGE": "Estado civil (1 casado, 2 soltero, 3 otro)", "AGE": "Edad (años)",
    **{f"PAY_{n}": f"Estado de pago, {m} (-2/-1/0 al día; 1 a 9 = meses de atraso)"
       for n, m in (("0", "septiembre"), ("2", "agosto"), ("3", "julio"), ("4", "junio"), ("5", "mayo"), ("6", "abril"))},
    **{f"BILL_AMT{n}": f"Monto de la factura, {m} (NT$)"
       for n, m in (("1", "septiembre"), ("2", "agosto"), ("3", "julio"), ("4", "junio"), ("5", "mayo"), ("6", "abril"))},
    **{f"PAY_AMT{n}": f"Pago realizado, {m} (NT$)"
       for n, m in (("1", "septiembre"), ("2", "agosto"), ("3", "julio"), ("4", "junio"), ("5", "mayo"), ("6", "abril"))},
}
APP_RESPONSIBLE_USE = "\n".join([
    "- Esta herramienta **apoya** una decisión de seguimiento preventivo; la decisión final es de una persona del área de riesgo.",
    "- Se entrenó con datos de un banco de Taiwán de 2005 (UCI, CC BY 4.0): no la uses en otro país o época sin validarla de nuevo.",
    "- El umbral de marcado y la relación de costo 5 a 1 son **hipótesis del grupo**, no costos medidos del banco.",
    "- **No** la uses para negar crédito de forma automática ni para decidir por sexo, edad o estado civil.",
    "- Revisa los límites en `models/ficha_modelo.md`.",
])


def _table(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="index")


def validate_and_freeze(train, *, revision, versions):
    """07: valida con el entrenamiento y congela calibración y umbral ANTES de abrir la prueba."""
    controls = v.ExecutionControls(fit_real=True, calibrate_real=True)
    decision = v.provisional_decision(revision=revision, versions=versions, training_sha256=TRAIN_SHA256)
    internal = v.evaluate_internal(train, controls=controls, n_splits=5, seed=42, reserve_seed=7, n_boot=1000)
    settings = {}
    for name in (v.ORIGINAL, v.REFERENCE):
        # Regla escrita antes: calibrar solo si baja Brier y log loss sin bajar el AUC OOF.
        sigmoid = internal["recommendations"][name]["recommend_sigmoid"]
        label = name + (" + sigmoid" if sigmoid else "")
        settings[name] = {"calibration": "sigmoid" if sigmoid else "none",
                          "threshold": float(internal["thresholds"][label]["threshold"])}
    frozen = v.freeze_for_evaluation(decision, settings, approved=True, candidates=(v.ORIGINAL, v.REFERENCE))
    importance = v.interpretation(internal["models"][v.ORIGINAL], internal["reserve"], controls=controls,
                                  n_repeats=5, seed=42)
    selected = v.ORIGINAL + (" + sigmoid" if settings[v.ORIGINAL]["calibration"] == "sigmoid" else "")
    segments = v.segment_metrics(internal["reserve"], internal["probabilities"][selected],
                                 threshold=settings[v.ORIGINAL]["threshold"])
    report = v.validation_report(frozen, controls=controls, internal=internal, importance=importance,
                                 segments=segments)
    return {"frozen": frozen, "internal": internal, "settings": settings, "importance": importance,
            "segments": segments, "report": report}


def prepare_authorized_handoff(frozen, report, destination, *, approved_by):
    """Firma el traspaso 07->08 (responsable y fecha UTC) y lo vuelve a leer verificando su huella."""
    approved_at = datetime.now(timezone.utc).isoformat()
    handoff = prepare_handoff(frozen, report, authorize_final_fit=True, approved_by=approved_by,
                              approved_at=approved_at)
    receipt = write_handoff(handoff, destination)
    return read_handoff(receipt["path"], trusted=True, expected_sha256=receipt["sha256"]), receipt, approved_at


def definitive_approval(handoff, *, approved_by):
    """Aprobación definitiva ligada al informe y a la configuración revisados. La da una persona, no las métricas."""
    record = {"approved": True, "approved_by": approved_by,
              "approved_at": datetime.now(timezone.utc).isoformat(),
              "validation_report_id": handoff["handoff"]["validation_report"]["report_id"],
              "configuration_sha256": handoff["handoff"]["configuration_sha256"]}
    return approve_for_inference(handoff, approved=True, approval=record)


def fit_and_save_bundle(train, decision, directory, *, revision):
    """08: único ajuste final con los 24.000 clientes + paquete nuevo (nunca sobrescribe) y su recuperación."""
    controls = PersistenceControls(train_real=True, publish_real_artifacts=True)
    model = train_final_reference(train, decision=decision, controls=controls)
    inputs, _ = make_xy(train)
    saved = save_bundle(model, inputs, directory, model_name=MODEL_NAME, model_version="candidate-08",
                        decision=decision, controls=controls, revision=revision,
                        validation_metrics=selected_validation_metrics(decision))
    bundle = load_bundle(saved["directory"], trusted=True, expected_manifest_sha256=saved["manifest_sha256"],
                         require_approved=decision["status"] == "approved_definitive")
    compare_predictions(model, bundle.model, inputs.head(500), bundle.schema)  # ida y vuelta idéntica
    return bundle, saved, inputs


def build_app_schema(bundle_schema, threshold=OPERATING_THRESHOLD):
    """Esquema de la app = esquema del paquete + umbral de trabajo explícito (sin tocar el paquete verificado)."""
    if isinstance(threshold, bool) or not 0 < float(threshold) < 1:
        raise ValueError("El umbral debe estar entre 0 y 1, sin los extremos")
    schema = deepcopy(bundle_schema)
    schema.update(positive_class="1", decision_threshold=float(threshold), decision_rule=OPERATING_RULE,
                  app_title=APP_TITLE, labels=dict(APP_LABELS), responsible_use=APP_RESPONSIBLE_USE)
    return schema


def export_for_app(bundle_directory, models_dir, threshold=OPERATING_THRESHOLD):
    """Copia exacta de los bytes verificados + esquema con umbral. La app NO consulta Databricks."""
    models_dir = Path(models_dir)
    shutil.copyfile(Path(bundle_directory) / "pipeline.joblib", models_dir / "model.joblib")
    schema = json.loads((Path(bundle_directory) / "input_schema.json").read_text(encoding="utf-8"))
    (models_dir / "input_schema.json").write_text(
        json.dumps(build_app_schema(schema, threshold), indent=2, ensure_ascii=False), encoding="utf-8")


def register_locally(bundle, inputs, *, project_root, experiment_name="riskvision-08-registro"):
    """08: sube evidencia, registra la versión y asigna el alias champion en el destino que devuelva configure_mlflow."""
    import mlflow
    from proyecto.tracking.mlflow_utils import configure_mlflow, registered_model_name, set_experiment
    from proyecto.tracking.model_registry import (promote_model_version, publish_bundle_artifacts,
                                                  register_definitive_model)
    controls = PersistenceControls(train_real=True, register_definitive=True, promote_or_approve=True,
                                   publish_real_artifacts=True)
    where = configure_mlflow()
    experiment_path = set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id
    tracking = publish_bundle_artifacts(bundle, experiment_id=experiment_id, controls=controls)
    registered = register_definitive_model(bundle, inputs.head(20), registered_name=registered_model_name(MODEL_NAME),
                                           experiment_id=experiment_id, source_code_dir=Path(project_root) / "src/proyecto",
                                           controls=controls)
    promoted = promote_model_version(bundle, registered, controls=controls)
    return {"destino": where, "experimento": experiment_path, "tracking": tracking, "registro": promoted}


def evaluate_test_once(train, test_path, bundle, handoff):
    """07b: abre la prueba UNA sola vez. Ya estaba expuesta (EDA y torneo): evaluación complementaria."""
    controls = v.ExecutionControls(fit_real=True, calibrate_real=False, evaluate_test=True)
    test = v.load_complementary_test(test_path, controls=controls, decision=handoff, training_ids=set(train[ID_COL]))
    reference = v.fit_training_candidate(train, v.candidate_factories(training_sha256=TRAIN_SHA256)[v.REFERENCE],
                                         controls=controls, calibrated=False)
    complementary = v.evaluate_complementary({v.ORIGINAL: bundle.model, v.REFERENCE: reference}, test,
                                             controls=controls, decision=handoff, n_boot=1000, seed=42)
    return complementary, test


def check_inference(models_dir, sample_inputs):
    """09: el modelo exportado puntúa, respeta el umbral y rechaza una edad imposible."""
    from proyecto.inference.predict import InputValidationError, predict_with_scores
    models_dir = Path(models_dir)
    model = joblib.load(models_dir / "model.joblib")
    schema = json.loads((models_dir / "input_schema.json").read_text(encoding="utf-8"))
    scored = predict_with_scores(sample_inputs, model, schema)
    coherent = bool((scored["decision"].to_numpy() == np.where(scored["proba_1"] >= schema["decision_threshold"], "1", "0")).all())
    invalid = sample_inputs.head(1).copy()
    invalid["AGE"] = 5
    try:
        predict_with_scores(invalid, model, schema)
        rejected = None
    except (InputValidationError, ValueError) as error:
        rejected = str(error)[:200]
    return {"scored": scored, "decision_matches_threshold": coherent, "invalid_input_rejected": rejected}
