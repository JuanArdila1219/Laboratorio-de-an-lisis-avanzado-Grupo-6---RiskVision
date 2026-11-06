"""Publicación explícita: Tracking y Model Registry son operaciones distintas."""
from pathlib import Path

from proyecto.models.persistence import (FILES, PersistenceControls, prediction_check,
                                        require_approved_model, verify_bundle)


def _publish_guard(bundle, controls):
    if not controls.publish_real_artifacts:
        raise PermissionError("Publicación real bloqueada")
    if bundle.metadata["data_scope"] != "real":
        raise PermissionError("Un artefacto sintético no se publica en servicios remotos")
    verify_bundle(bundle.directory, expected_manifest_sha256=bundle.manifest_sha256)


def publish_bundle_artifacts(bundle, *, experiment_id, controls=PersistenceControls(), client=None):
    """Crea un run nuevo; admite provisionales, sin crear versiones ni alias."""
    _publish_guard(bundle, controls)
    if not experiment_id:
        raise ValueError("Se requiere un experimento explícito")
    if client is None:
        from mlflow import MlflowClient
        client = MlflowClient()
    run = client.create_run(str(experiment_id), tags={"riskvision.stage": "08",
        "riskvision.status": bundle.metadata["status"], "riskvision.role": "artifact_tracking",
        "riskvision.manifest_sha256": bundle.manifest_sha256})
    run_id = run.info.run_id
    try:
        for name in (*FILES, "manifest.json"):
            client.log_artifact(run_id, str(Path(bundle.directory) / name), artifact_path="riskvision_bundle")
        for name, value in (bundle.metadata.get("validation_metrics") or {}).items():
            client.log_metric(run_id, name, float(value))
        client.set_terminated(run_id, status="FINISHED")
    except BaseException:
        client.set_terminated(run_id, status="FAILED")
        raise
    return {"scope": "tracking_artifacts", "run_id": run_id, "experiment_id": str(experiment_id),
            "manifest_sha256": bundle.manifest_sha256, "registered_version": None}


def register_definitive_model(bundle, input_example, *, registered_name, experiment_id,
                              source_code_dir, controls=PersistenceControls(), mlflow_module=None):
    """Publica flavor sklearn y registra SU versión exacta; nunca asigna un alias.

    code_paths empaqueta proyecto para importar los selectores del pipeline.
    La aprobación debe existir antes de invocar esta función.
    """
    if not controls.register_definitive:
        raise PermissionError("Registro definitivo bloqueado")
    require_approved_model(bundle.metadata)
    _publish_guard(bundle, controls)
    prediction_check(bundle.model, input_example, bundle.schema)
    source = Path(source_code_dir).resolve()
    if source.name != "proyecto" or not (source / "models/train.py").is_file():
        raise ValueError("source_code_dir debe señalar src/proyecto importable")
    if not experiment_id or not registered_name or "/" in registered_name:
        raise ValueError("Nombre de modelo y experimento explícitos requeridos")
    if mlflow_module is None:
        import mlflow as mlflow_module
    ml = mlflow_module
    if ml.active_run() is not None:
        raise ValueError("Cerrar el run activo antes de registrar")
    if ml.get_registry_uri().startswith("databricks-uc") and len(registered_name.split(".")) != 3:
        raise ValueError("Unity Catalog requiere catalog.schema.model explícitos")
    example = input_example.loc[:, bundle.schema["input_columns"]].copy()
    # MLflow infiere mejor tipos primitivos que pandas.Categorical; no cambia valores.
    for name in example.select_dtypes(include="category"):
        example[name] = example[name].astype(float)
    prediction_check(bundle.model, example, bundle.schema)
    signature = ml.models.infer_signature(example, bundle.model.predict_proba(example))
    requirements = [f"{name}=={value}" for name, value in bundle.metadata["versions"].items() if name != "python"]
    with ml.start_run(experiment_id=str(experiment_id), tags={"riskvision.stage": "08",
            "riskvision.status": "approved_definitive", "riskvision.manifest_sha256": bundle.manifest_sha256}) as run:
        info = ml.sklearn.log_model(sk_model=bundle.model, name="model", signature=signature,
            input_example=example, code_paths=[str(source)], serialization_format="cloudpickle",
            pip_requirements=requirements, pyfunc_predict_fn="predict_proba")
        for name in (*FILES, "manifest.json"):
            ml.log_artifact(str(Path(bundle.directory) / name), artifact_path="riskvision_bundle")
        registered = ml.register_model(info.model_uri, registered_name)
        client = ml.MlflowClient()
        for key, value in {"riskvision.status": "approved_definitive",
                           "riskvision.manifest_sha256": bundle.manifest_sha256}.items():
            client.set_model_version_tag(registered_name, str(registered.version), key, value)
        return {"scope": "model_registry", "name": registered_name, "version": str(registered.version),
                "run_id": run.info.run_id, "model_uri": info.model_uri,
                "manifest_sha256": bundle.manifest_sha256}


def promote_model_version(bundle, receipt, *, controls=PersistenceControls(), alias="champion", client=None):
    if not controls.promote_or_approve:
        raise PermissionError("Promoción bloqueada")
    require_approved_model(bundle.metadata)
    _publish_guard(bundle, controls)
    if receipt.get("scope") != "model_registry" or receipt.get("manifest_sha256") != bundle.manifest_sha256:
        raise ValueError("La versión no corresponde al paquete aprobado")
    if client is None:
        from mlflow import MlflowClient
        client = MlflowClient()
    registered = client.get_model_version(receipt["name"], receipt["version"])
    if (registered.tags.get("riskvision.status") != "approved_definitive"
            or registered.tags.get("riskvision.manifest_sha256") != bundle.manifest_sha256):
        raise ValueError("La versión remota no coincide con la evidencia aprobada")
    client.set_registered_model_alias(receipt["name"], alias, receipt["version"])
    return {**receipt, "alias": alias}
