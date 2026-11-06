"""Paquetes Joblib verificables para 08/09, sin aprobación ni publicación implícitas.

SHA-256 detecta modificaciones, no acredita por sí solo la autenticidad de una
fuente. Joblib usa pickle: cargar únicamente paquetes de procedencia confiable.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from proyecto.config import CATEGORICAL, ROOT, ID_COL, TARGET
from proyecto.data.cleaning import PAY_COLS, PAY_DOMAIN, PREDICTORS, NON_NEGATIVE, _validate_predictors
from proyecto.inference.predict import InputValidationError, validate_inputs
from proyecto.inference.schema import build_schema
from proyecto.features.build import PACKAGE_E
from proyecto.models.candidates import LGBM_PARAMS, make_lgbm
from proyecto.models.decision import validate_handoff
from proyecto.models.validation import ExecutionControls as ValidationControls, ORIGINAL, fit_training_candidate
from proyecto.tracking.reporting import code_revision

FILES = ("pipeline.joblib", "input_schema.json", "metadata.json")
DOMAINS = {"SEX": [1, 2], "EDUCATION": [1, 2, 3, 4], "MARRIAGE": [1, 2, 3]}
PACKAGES = ("pandas", "numpy", "scipy", "scikit-learn", "joblib", "lightgbm", "xgboost", "mlflow")


@dataclass(frozen=True)
class PersistenceControls:
    train_real: bool = False
    register_definitive: bool = False
    promote_or_approve: bool = False
    publish_real_artifacts: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("Los permisos deben ser booleanos explícitos")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def dependency_versions():
    result = {"python": ".".join(str(value) for value in sys.version_info[:3])}
    for name in PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Falta {name} para documentar el entorno") from error
    return result


def _fitted_pipelines(model):
    check_is_fitted(model)
    pipelines = ([item.estimator for item in model.calibrated_classifiers_]
                 if isinstance(model, CalibratedClassifierCV) else [model])
    for pipeline in pipelines:
        if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["features", "prep", "model"]:
            raise ValueError("Debe persistirse el pipeline completo, no un estimador aislado")
        check_is_fitted(pipeline.named_steps["prep"])
        check_is_fitted(pipeline.named_steps["model"])
        if tuple(pipeline.named_steps["features"].kw_args["names"]) != PACKAGE_E:
            raise ValueError("Las características no coinciden con el paquete E declarado")
        if pipeline.named_steps["features"].kw_args.get("drop"):
            raise ValueError("El contrato conserva todos los predictores originales")
    if list(np.asarray(model.classes_)) != [0, 1]:
        raise ValueError("Se requieren clases [0, 1] en ese orden")
    return pipelines


def _check_columns(frame, columns):
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates or frame.empty:
        raise InputValidationError("Entrada vacía, inválida o con columnas duplicadas")
    if set(frame.columns) != set(columns):
        raise InputValidationError("Se requieren exactamente los 23 predictores originales, sin ID, objetivo ni derivadas")


def _validate_handoff_pipeline(model, schema, decision):
    validate_handoff(decision)
    calibrated = isinstance(model, CalibratedClassifierCV)
    expected = decision["settings"][ORIGINAL]["calibration"]
    if (calibrated != (expected == "sigmoid")
            or (calibrated and (model.method != "sigmoid" or model.ensemble is not False))):
        raise ValueError("La calibración del pipeline no coincide con la decisión autorizada")
    if schema["input_columns"] != decision["handoff"]["input_columns"]:
        raise ValueError("El orden del esquema no coincide con el contrato 07→08")


def _validate_cleaned_values(frame):
    try:
        _validate_predictors(frame)
        for name, allowed in DOMAINS.items():
            if not frame[name].dropna().isin(allowed).all():
                raise ValueError(f"{name}: se requieren códigos limpios del contrato {allowed}")
        if not frame["AGE"].dropna().between(18, 100).all():
            raise ValueError("AGE: fuera de 18–100")
        if not (frame["LIMIT_BAL"].dropna() > 0).all():
            raise ValueError("LIMIT_BAL debe ser positivo")
        if any((frame[name].dropna() < 0).any() for name in NON_NEGATIVE):
            raise ValueError("Los pagos no pueden ser negativos; las facturas sí pueden serlo")
    except ValueError as error:
        raise InputValidationError(str(error)) from error


def create_input_schema(training_inputs):
    """Entradas originales ya limpias; no aprende imputación ni transforma valores."""
    _check_columns(training_inputs, PREDICTORS)
    _validate_cleaned_values(training_inputs)
    schema = build_schema(training_inputs, [name for name in training_inputs if name not in CATEGORICAL],
                          list(CATEGORICAL), ["0", "1"], valid_range={"AGE": (18, 100)})
    schema.update(format_version=1, input_columns=list(training_inputs.columns),
                  contract="cleaned_original_predictors_v1", missing_values="pipeline_imputation",
                  categorical_domains=deepcopy(DOMAINS), pay_domain=sorted(PAY_DOMAIN),
                  decision_rule="probabilities_and_model_class_only; operating_threshold_not_approved_here")
    return schema


def _check_schema(schema):
    if (schema.get("format_version") != 1 or schema.get("contract") != "cleaned_original_predictors_v1"
            or len(schema.get("input_columns", [])) != len(PREDICTORS)
            or set(schema.get("input_columns", [])) != set(PREDICTORS)
            or set(schema.get("categorical", [])) != set(CATEGORICAL)
            or set(schema.get("numeric", [])) != set(PREDICTORS) - set(CATEGORICAL)
            or schema.get("class_names") != ["0", "1"]
            or schema.get("categorical_domains") != DOMAINS):
        raise ValueError("Esquema incompatible con el contrato de RiskVision")
    if schema.get("cost_matrix") is not None:
        raise ValueError("No se introduce una regla operativa de costos implícita en persistencia")


def validate_input_schema(frame, schema):
    _check_schema(schema)
    _check_columns(frame, schema["input_columns"])
    _validate_cleaned_values(frame)
    return validate_inputs(frame, schema)  # categorías válidas ausentes del ajuste solo generan aviso


def prediction_check(model, frame, schema):
    warnings = validate_input_schema(frame, schema)
    ordered = frame.loc[:, schema["input_columns"]]
    probability, predicted = np.asarray(model.predict_proba(ordered)), np.asarray(model.predict(ordered))
    if (probability.shape != (len(frame), 2) or not np.isfinite(probability).all()
            or not ((probability >= 0) & (probability <= 1)).all()
            or not np.allclose(probability.sum(axis=1), 1, atol=1e-12)
            or predicted.shape != (len(frame),) or not np.isin(predicted, [0, 1]).all()):
        raise ValueError("El modelo no genera clases/probabilidades válidas")
    return {"rows": len(frame), "class_order": [0, 1], "warnings": warnings}


def compare_predictions(reference, recovered, frame, schema):
    prediction_check(reference, frame, schema)
    prediction_check(recovered, frame, schema)
    X = frame.loc[:, schema["input_columns"]]
    np.testing.assert_array_equal(reference.predict(X), recovered.predict(X))
    np.testing.assert_allclose(reference.predict_proba(X), recovered.predict_proba(X), rtol=1e-14, atol=1e-15)
    return {"classes_equal": True, "probabilities_equivalent": True, "rtol": 1e-14, "atol": 1e-15}


def require_approved_model(metadata):
    """La aprobación es externa; este módulo nunca convierte provisional en definitivo."""
    decision = metadata.get("decision", {})
    approval = decision.get("definitive_approval", {})
    if (metadata.get("data_scope") != "real" or metadata.get("status") != "approved_definitive"
            or decision.get("status") != "approved_definitive" or not decision.get("frozen_for_evaluation")
            or approval.get("approved") is not True
            or not all(approval.get(key) for key in ("approved_by", "approved_at", "validation_report_id"))):
        raise PermissionError("El modelo no tiene aprobación definitiva documentada; provisional/sintético no es definitivo")
    if "handoff" in decision:
        validate_handoff(decision)
        if approval.get("configuration_sha256") != decision["handoff"]["configuration_sha256"]:
            raise PermissionError("La aprobación definitiva no corresponde a la configuración autorizada")


def save_bundle(model, training_inputs, directory, *, model_name, model_version, decision,
                controls=PersistenceControls(), data_scope="real", validation_metrics=None,
                experiment_id=None, revision=None):
    """Crea una carpeta nueva con cuatro archivos; nunca sobrescribe otra versión.

    Solo la opción explícita synthetic permite guardar sin permiso de artefactos
    reales, y únicamente bajo el directorio temporal del sistema.
    """
    directory = Path(directory).resolve()
    if data_scope not in {"synthetic", "real"}:
        raise ValueError("data_scope debe ser real o synthetic")
    if data_scope == "real" and not controls.publish_real_artifacts:
        raise PermissionError("Persistencia real bloqueada: ALLOW_REAL_ARTIFACT_PUBLICATION=False")
    if data_scope == "synthetic":
        if not directory.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise PermissionError("Los modelos sintéticos se guardan solo en ubicaciones temporales")
        if decision.get("status") != "provisional":
            raise PermissionError("Una prueba sintética no puede declararse modelo definitivo")
    for value in (model_name, model_version):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
            raise ValueError("Nombre/versión inválidos")
    if decision.get("status") not in {"provisional", "approved_definitive"} or decision.get("features") != list(PACKAGE_E):
        raise ValueError("Decisión o paquete de variables inválidos")
    pipelines = _fitted_pipelines(model)
    base = pipelines[0].named_steps["model"]
    actual_params = base.get_params()
    if not decision.get("params") or any(actual_params.get(key) != value for key, value in decision["params"].items()):
        raise ValueError("Los parámetros del pipeline no coinciden con su decisión")
    schema = create_input_schema(training_inputs)
    if "handoff" in decision:
        _validate_handoff_pipeline(model, schema, decision)
    if list(model.feature_names_in_) != schema["input_columns"]:
        raise ValueError("El orden de entrada no coincide con el utilizado en el ajuste")
    smoke = prediction_check(model, training_inputs.head(10), schema)
    buffer = BytesIO()
    joblib.dump(model, buffer)
    payload = buffer.getvalue()
    schema_bytes = _json_bytes(schema)
    metadata = {"format_version": 1, "artifact_id": uuid4().hex,
        "model_name": model_name, "model_version": model_version, "registered_version": None,
        "status": decision["status"], "data_scope": data_scope,
        "created_at": datetime.now(timezone.utc).isoformat(), "estimator_class": type(base).__name__,
        "wrapper_class": type(model).__name__, "features": list(PACKAGE_E), "hyperparameters": actual_params,
        "code": dict(revision if revision is not None else code_revision(ROOT)), "versions": dependency_versions(),
        "artifact_sha256": _sha(payload), "schema_sha256": _sha(schema_bytes),
        "mlflow_experiment_id": None if experiment_id is None else str(experiment_id),
        "validation_metrics": deepcopy(validation_metrics), "decision": deepcopy(decision), "prediction_check": smoke}
    if metadata["status"] == "approved_definitive":
        require_approved_model(metadata)
    if data_scope == "real" and not all(metadata["code"].get(key) for key in ("branch", "commit")):
        raise ValueError("La persistencia real requiere rama y commit")
    contents = {"pipeline.joblib": payload, "input_schema.json": schema_bytes, "metadata.json": _json_bytes(metadata)}
    manifest = {"format_version": 1, "artifact_id": metadata["artifact_id"],
                "files": {name: _sha(content) for name, content in contents.items()}}
    manifest_bytes = _json_bytes(manifest)
    directory.mkdir(parents=True, exist_ok=False)
    # Un fallo deja el paquete incompleto, verificablemente inválido; nunca se publica como válido.
    for name, content in {**contents, "manifest.json": manifest_bytes}.items():
        with (directory / name).open("xb") as stream:
            stream.write(content)
    return {"directory": str(directory), "artifact_sha256": metadata["artifact_sha256"],
            "manifest_sha256": _sha(manifest_bytes), "artifact_id": metadata["artifact_id"], "status": metadata["status"]}


def _verified_contents(directory, expected_manifest_sha256=None):
    directory = Path(directory)
    for name in (*FILES, "manifest.json"):
        if (directory / name).is_symlink() or not (directory / name).is_file():
            raise ValueError("Paquete incompleto o con enlaces no admitidos")
    manifest_bytes = (directory / "manifest.json").read_bytes()
    if expected_manifest_sha256 is not None and _sha(manifest_bytes) != expected_manifest_sha256:
        raise ValueError("SHA-256 del manifiesto no coincide con la referencia confiable")
    manifest = json.loads(manifest_bytes)
    if manifest.get("format_version") != 1 or set(manifest.get("files", {})) != set(FILES):
        raise ValueError("Formato de manifiesto inválido")
    contents = {name: (directory / name).read_bytes() for name in FILES}
    if any(_sha(payload) != manifest["files"][name] for name, payload in contents.items()):
        raise ValueError("Integridad SHA-256 inválida: se modificó un archivo")
    metadata, schema = json.loads(contents["metadata.json"]), json.loads(contents["input_schema.json"])
    if (metadata.get("format_version") != 1 or metadata.get("artifact_id") != manifest.get("artifact_id")
            or metadata.get("status") not in {"provisional", "approved_definitive"}
            or metadata.get("data_scope") not in {"synthetic", "real"}
            or metadata.get("decision", {}).get("status") != metadata.get("status")
            or metadata.get("features") != list(PACKAGE_E)
            or metadata.get("artifact_sha256") != manifest["files"]["pipeline.joblib"]
            or metadata.get("schema_sha256") != manifest["files"]["input_schema.json"]):
        raise ValueError("Metadatos inconsistentes con el manifiesto")
    _check_schema(schema)
    return {"metadata": metadata, "schema": schema, "manifest": manifest,
            "manifest_sha256": _sha(manifest_bytes), "payload": contents["pipeline.joblib"]}


def verify_bundle(directory, *, expected_manifest_sha256=None):
    verified = _verified_contents(directory, expected_manifest_sha256)
    return {key: value for key, value in verified.items() if key != "payload"}  # no deserializa código


@dataclass(frozen=True)
class ModelBundle:
    model: object
    schema: dict
    metadata: dict
    manifest_sha256: str
    directory: str

    @property
    def classes_(self):
        return self.model.classes_

    def predict_proba(self, frame):
        validate_input_schema(frame, self.schema)
        return self.model.predict_proba(frame.loc[:, self.schema["input_columns"]])

    def predict(self, frame):
        validate_input_schema(frame, self.schema)
        return self.model.predict(frame.loc[:, self.schema["input_columns"]])


def load_bundle(directory, *, trusted=False, expected_manifest_sha256=None, require_approved=False):
    if trusted is not True:
        raise PermissionError("Joblib requiere confirmar una procedencia confiable")
    verified = _verified_contents(directory, expected_manifest_sha256)
    metadata = verified["metadata"]
    if metadata["data_scope"] == "real" and expected_manifest_sha256 is None:
        raise ValueError("Un artefacto real requiere la huella externa de su manifiesto")
    if require_approved:
        require_approved_model(metadata)
    current = dependency_versions()
    if (any(metadata["versions"].get(name) != current[name] for name in PACKAGES)
            or metadata["versions"]["python"].split(".")[:2] != current["python"].split(".")[:2]):
        raise RuntimeError("Dependencias incompatibles con el artefacto; reproducir su entorno antes de cargar")
    model = joblib.load(BytesIO(verified["payload"]))  # exactamente los bytes verificados
    pipelines = _fitted_pipelines(model)
    if _json_bytes(pipelines[0].named_steps["model"].get_params()) != _json_bytes(metadata["hyperparameters"]):
        raise ValueError("Modelo y parámetros documentados no corresponden")
    if list(model.feature_names_in_) != verified["schema"]["input_columns"]:
        raise ValueError("Modelo y esquema no corresponden")
    if "handoff" in metadata["decision"]:
        _validate_handoff_pipeline(model, verified["schema"], metadata["decision"])
    return ModelBundle(model, verified["schema"], metadata, verified["manifest_sha256"], str(Path(directory).resolve()))


def train_final_reference(frame, *, decision, controls=PersistenceControls()):
    if not controls.train_real:
        raise PermissionError("Entrenamiento final bloqueado: ALLOW_FINAL_REAL_TRAINING=False")
    validate_handoff(decision or {})
    if (not decision or not decision.get("frozen_for_evaluation") or decision.get("params") != LGBM_PARAMS
            or decision.get("features") != list(PACKAGE_E) or decision.get("model") != ORIGINAL):
        raise PermissionError("Falta configuración congelada de la referencia original")
    if not re.fullmatch(r"[a-f0-9]{64}", decision.get("training_sha256", "")):
        raise ValueError("Falta una huella SHA-256 del entrenamiento")
    if frame.attrs.get("source_sha256") != decision.get("training_sha256"):
        raise ValueError("Entrenamiento distinto al de la decisión")
    if tuple(frame.shape) != (24000, 25):
        raise ValueError("El ajuste final de 08 requiere los 24.000 registros y 25 columnas")
    if list(frame.drop(columns=[ID_COL, TARGET])) != decision["handoff"]["input_columns"]:
        raise ValueError("El orden de predictores no coincide con el contrato")
    current = dependency_versions()
    if any(decision["versions"].get(name) != current[name] for name in PACKAGES):
        raise RuntimeError("El entorno final difiere del validado en 07")
    if decision["versions"].get("python", "").split(".")[:2] != current["python"].split(".")[:2]:
        raise RuntimeError("La versión Python final difiere de la validada en 07")
    choice = decision.get("settings", {}).get(decision["model"], {})
    if choice.get("calibration") not in {"none", "sigmoid"}:
        raise ValueError("Calibración no definida")
    calibrated = choice["calibration"] == "sigmoid"
    return fit_training_candidate(frame, partial_reference(), controls=ValidationControls(
        fit_real=True, calibrate_real=calibrated), calibrated=calibrated)


def partial_reference():
    from functools import partial
    return partial(make_lgbm, PACKAGE_E)


def synthetic_persistence_check(directory):
    """Demostración local reproducible; no lee datos ni publica registros."""
    rng, n = np.random.default_rng(42), 200
    X = pd.DataFrame({"LIMIT_BAL": rng.integers(10000, 500000, n), "SEX": rng.integers(1, 3, n),
                      "EDUCATION": rng.integers(1, 5, n), "MARRIAGE": rng.integers(1, 4, n),
                      "AGE": rng.integers(21, 70, n)})
    for name in PAY_COLS:
        X[name] = rng.integers(-2, 4, n)
    for i in range(1, 7):
        X[f"BILL_AMT{i}"] = rng.integers(-1000, 200000, n)
        X[f"PAY_AMT{i}"] = rng.integers(0, 20000, n)
    X = X.loc[:, PREDICTORS]
    y = (rng.random(n) < 1 / (1 + np.exp(-(X["PAY_0"] - 1)))).astype(int)
    model = make_lgbm(PACKAGE_E).fit(X.iloc[:150], y[:150])
    decision = {"status": "provisional", "model": "LightGBM original", "features": list(PACKAGE_E),
                "params": deepcopy(LGBM_PARAMS), "provenance": "synthetic persistence demonstration, no real validation"}
    receipt = save_bundle(model, X.iloc[:150], directory, model_name="riskvision_lightgbm_e",
                          model_version="synthetic-08", decision=decision, data_scope="synthetic")
    recovered = load_bundle(directory, trusted=True, expected_manifest_sha256=receipt["manifest_sha256"])
    return {"scope": "synthetic", "status": "provisional", "receipt": receipt,
            "check": compare_predictions(model, recovered, X.iloc[150:], recovered.schema)}
