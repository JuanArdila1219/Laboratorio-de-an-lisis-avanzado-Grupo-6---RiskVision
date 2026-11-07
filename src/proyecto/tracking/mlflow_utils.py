"""Conexión y registro de ejecuciones en MLflow: Databricks si hay conexión, local si no.

Cómo usarlo en un notebook (ver `.env.example` para la conexión):
    from proyecto.tracking.mlflow_utils import configure_mlflow, set_experiment, log_run
    where = configure_mlflow()                 # 'databricks (...)' o 'local' (respaldo automático)
    set_experiment("mi-proyecto-optimizacion")  # en Databricks va a EXPERIMENT_DIR si lo definiste
    log_run("ensayo-1", {"depth": 6}, {"cost": 0.48}, tags={"etapa": "06_optimizacion"})
Si no hay conexión (o pones MLFLOW_FORCE_LOCAL=1 en .env), todo se guarda en `mlflow_local.db`.

Dos formas de conectar a Databricks, ambas leídas de `.env`:
  - PROFILE=<perfil de ~/.databrickscfg>        -> databricks://<perfil>
  - DATABRICKS_HOST + DATABRICKS_TOKEN          -> databricks
"""
import os
import warnings
from pathlib import Path

import mlflow
from dotenv import load_dotenv

# src/proyecto/tracking/mlflow_utils.py -> la raíz del repositorio está tres niveles arriba de `proyecto`
ROOT = Path(__file__).resolve().parents[3]
LOCAL_DB = ROOT / "mlflow_local.db"  # se genera sola; no se versiona
PLACEHOLDER_PROFILE = "nombre-de-tu-perfil"  # valor de .env.example sin completar


def is_databricks() -> bool:
    """True si el código corre dentro de un workspace de Databricks."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _reachable() -> bool:
    try:
        mlflow.search_experiments(max_results=1)
        return True
    except Exception as error:  # sin red, credenciales inválidas, cuota agotada, CLI fuera del PATH...
        reason = str(error).splitlines()[0][:140]
        warnings.warn(f"Databricks no disponible ({reason}). Se usa MLflow local.", stacklevel=3)
        return False


def configure_mlflow(force_local: bool = False, local_db: Path = LOCAL_DB, env_file: Path = ROOT / ".env",
                     *, backend: str = "auto") -> str:
    """Configura el destino sin crear experimentos.

    auto conserva perfiles y respaldo local fuera del workspace. Dentro del
    workspace, o con backend=databricks, comprueba el servidor nativo y propaga
    errores sin respaldo local. backend=local equivale a force_local=True.
    """
    if backend not in {"auto", "local", "databricks"}:
        raise ValueError("backend debe ser auto, local o databricks")
    if force_local and backend == "databricks":
        raise ValueError("force_local es incompatible con backend=databricks")
    force_local = force_local or backend == "local"
    if (backend == "databricks" or is_databricks()) and not force_local:
        # Restablece un posible URI SQLite de una celda anterior. La autenticación
        # dentro del workspace es nativa; no se lee .env ni se cae a local.
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        mlflow.search_experiments(max_results=1)
        return "databricks (dentro del workspace)"
    load_dotenv(env_file)
    if not force_local and os.environ.get("MLFLOW_FORCE_LOCAL") != "1":
        profile = os.environ.get("PROFILE")
        if profile and profile != PLACEHOLDER_PROFILE:
            mlflow.set_tracking_uri(f"databricks://{profile}")
            mlflow.set_registry_uri(f"databricks-uc://{profile}")
            if _reachable():
                return f"databricks (perfil {profile})"
        elif os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"):
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
            if _reachable():
                return "databricks (token)"
    local_uri = f"sqlite:///{Path(local_db).as_posix()}"
    mlflow.set_tracking_uri(local_uri)
    mlflow.set_registry_uri(local_uri)
    return "local"


def uses_databricks() -> bool:
    """True si MLflow quedó apuntando a Databricks (tras `configure_mlflow`)."""
    return mlflow.get_tracking_uri().startswith("databricks")


def _databricks_user() -> str:
    from databricks.sdk import WorkspaceClient

    profile = os.environ.get("PROFILE")
    if profile and profile != PLACEHOLDER_PROFILE and not is_databricks():
        return WorkspaceClient(profile=profile).current_user.me().user_name
    return WorkspaceClient().current_user.me().user_name


def resolve_experiment_name(name: str) -> str:
    """Resuelve la ruta sin crear ni activar experimentos."""
    if uses_databricks():
        if name.startswith("/"):
            return name
        folder = os.environ.get("EXPERIMENT_DIR", "").rstrip("/") or f"/Users/{_databricks_user()}"
        if not folder.startswith("/"):
            raise ValueError("EXPERIMENT_DIR debe ser una ruta absoluta del workspace")
        return f"{folder}/{name}"
    return name


def set_experiment(name: str) -> str:
    """Crea o activa el experimento. En Databricks el nombre debe ser una ruta.

    Si `.env` define EXPERIMENT_DIR (una carpeta del workspace que ya existe), el experimento va ahí;
    si no, en /Users/<usuario>/.
    """
    full_name = resolve_experiment_name(name)
    mlflow.set_experiment(full_name)
    return full_name


def registered_model_name(name: str) -> str:
    """Nombre completo para el registro: catalog.schema.name en Unity Catalog; el nombre solo en local."""
    if not mlflow.get_registry_uri().startswith("databricks"):
        return name
    catalog = os.environ.get("UC_CATALOG", "workspace")
    schema = os.environ.get("UC_SCHEMA", "default")
    return f"{catalog}.{schema}.{name}"


def log_run(run_name: str, params: dict, metrics: dict, tags: dict | None = None, nested: bool = False) -> str:
    """Registra una ejecución completa (parámetros, métricas y etiquetas) y devuelve su run_id."""
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.log_params({k: ("None" if v is None else v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        if tags:
            mlflow.set_tags(tags)
        return run.info.run_id


def log_tournament_runs(csv_path, tags: dict | None = None) -> list[dict]:
    """Registra UNA ejecución por candidato del torneo con las métricas por pliegue YA medidas (no recalcula nada).

    Es un registro a posteriori: las métricas salen de `reports/05_torneo_por_pliegue.csv`. Cada ejecución guarda la media
    de cada métrica y la desviación del ROC AUC entre pliegues. Debe llamarse después de `configure_mlflow()` y `set_experiment()`.
    """
    import pandas as pd

    table = pd.read_csv(csv_path)
    metric_columns = [c for c in table.columns if c not in {"modelo", "pliegue"}]
    receipts = []
    for model, group in table.groupby("modelo", sort=False):
        metrics = {f"mean_{c}": group[c].mean() for c in metric_columns}
        metrics["std_roc_auc"] = group["roc_auc"].std(ddof=1) if len(group) > 1 else 0.0
        run_id = log_run(f"torneo · {model}", {"modelo": model, "n_pliegues": len(group), "fuente": Path(csv_path).name},
                         metrics, tags={"etapa": "05_torneo", "registro": "a_posteriori", **(tags or {})})
        receipts.append({"modelo": model, "run_id": run_id, "mean_roc_auc": round(float(metrics["mean_roc_auc"]), 4)})
    return receipts
