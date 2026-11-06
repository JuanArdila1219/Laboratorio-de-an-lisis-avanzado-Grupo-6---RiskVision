import mlflow
import pytest

from proyecto.tracking import mlflow_utils as mu


@pytest.fixture(autouse=True)
def isolated_tracking(tmp_path, monkeypatch):
    """Evita bases, artefactos y credenciales reales durante las pruebas."""
    monkeypatch.chdir(tmp_path)
    for name in ("DATABRICKS_RUNTIME_VERSION", "PROFILE", "DATABRICKS_HOST",
                 "DATABRICKS_TOKEN", "MLFLOW_FORCE_LOCAL", "EXPERIMENT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_force_local_uses_sqlite_and_logs_a_run(tmp_path):
    where = mu.configure_mlflow(force_local=True, local_db=tmp_path / "mlflow.db", env_file=tmp_path / "no_existe.env")
    assert where == "local"
    assert mlflow.get_tracking_uri().startswith("sqlite:///")
    assert not mu.uses_databricks()

    mu.set_experiment("prueba-local")
    run_id = mu.log_run("r1", {"alpha": 0.1, "depth": None}, {"cost": 0.5}, tags={"etapa": "test"})
    run = mlflow.get_run(run_id)
    assert run.data.params == {"alpha": "0.1", "depth": "None"}
    assert run.data.metrics["cost"] == 0.5
    assert run.data.tags["etapa"] == "test"


def test_registered_model_name_is_plain_when_local(tmp_path):
    mu.configure_mlflow(force_local=True, local_db=tmp_path / "mlflow.db", env_file=tmp_path / "no_existe.env")
    assert mu.registered_model_name("mi_modelo") == "mi_modelo"


def test_missing_credentials_fall_back_to_local(tmp_path, monkeypatch):
    for var in ("PROFILE", "DATABRICKS_HOST", "DATABRICKS_TOKEN", "MLFLOW_FORCE_LOCAL"):
        monkeypatch.delenv(var, raising=False)
    where = mu.configure_mlflow(local_db=tmp_path / "mlflow.db", env_file=tmp_path / "no_existe.env")
    assert where == "local"


def test_experiment_dir_is_ignored_in_local_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_DIR", "/Users/alguien/carpeta")
    mu.configure_mlflow(force_local=True, local_db=tmp_path / "mlflow.db", env_file=tmp_path / "no_existe.env")
    assert mu.set_experiment("prueba") == "prueba"


@pytest.mark.parametrize("backend", ["auto", "databricks"])
def test_workspace_resets_local_uri_without_loading_env(backend, monkeypatch):
    calls = []
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "serverless")
    monkeypatch.setenv("MLFLOW_FORCE_LOCAL", "1")
    monkeypatch.setattr(mu, "load_dotenv", lambda *a: pytest.fail("No leer .env en workspace"))
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: calls.append(("tracking", uri)))
    monkeypatch.setattr(mlflow, "set_registry_uri", lambda uri: calls.append(("registry", uri)))
    monkeypatch.setattr(mlflow, "search_experiments", lambda **kw: calls.append(("check", kw)))
    assert mu.configure_mlflow(backend=backend).startswith("databricks")
    assert calls == [("tracking", "databricks"), ("registry", "databricks-uc"),
                     ("check", {"max_results": 1})]


def test_explicit_databricks_without_runtime_marker(monkeypatch):
    calls = []
    monkeypatch.setattr(mlflow, "set_tracking_uri", calls.append)
    monkeypatch.setattr(mlflow, "set_registry_uri", calls.append)
    monkeypatch.setattr(mlflow, "search_experiments", lambda **kw: [])
    assert mu.configure_mlflow(backend="databricks").startswith("databricks")
    assert calls == ["databricks", "databricks-uc"]


def test_workspace_failure_does_not_fall_back_to_sqlite(monkeypatch):
    calls = []
    monkeypatch.setattr(mlflow, "set_tracking_uri", calls.append)
    monkeypatch.setattr(mlflow, "set_registry_uri", lambda uri: None)
    def unavailable(**kwargs):
        raise RuntimeError("Servidor no disponible")
    monkeypatch.setattr(mlflow, "search_experiments", unavailable)
    with pytest.raises(RuntimeError, match="Servidor no disponible"):
        mu.configure_mlflow(backend="databricks")
    assert calls == ["databricks"]


@pytest.mark.parametrize("options", [{"backend": "otro"}, {"backend": "databricks", "force_local": True}])
def test_invalid_backend_is_rejected(options):
    with pytest.raises(ValueError):
        mu.configure_mlflow(**options)


def test_explicit_local_still_works_in_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "serverless")
    assert mu.configure_mlflow(backend="local", local_db=tmp_path / "local.db",
                               env_file=tmp_path / "missing.env") == "local"
    assert mlflow.get_tracking_uri().startswith("sqlite:///")


def test_full_experiment_name_is_idempotent(monkeypatch):
    monkeypatch.setattr(mu, "uses_databricks", lambda: True)
    monkeypatch.setenv("EXPERIMENT_DIR", "/Shared/riskvision/")
    calls = []
    monkeypatch.setattr(mlflow, "set_experiment", calls.append)
    full_name = mu.resolve_experiment_name("nuevo")
    assert full_name == "/Shared/riskvision/nuevo"
    assert mu.set_experiment(full_name) == full_name
    assert calls == [full_name]


def test_relative_experiment_folder_is_rejected(monkeypatch):
    monkeypatch.setattr(mu, "uses_databricks", lambda: True)
    monkeypatch.setenv("EXPERIMENT_DIR", "carpeta-relativa")
    with pytest.raises(ValueError, match="absoluta"):
        mu.resolve_experiment_name("nuevo")
