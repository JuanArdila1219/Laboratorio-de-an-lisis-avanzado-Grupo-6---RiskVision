"""Diagnóstico reproducible sin instalar paquetes ni abrir datos."""
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

from proyecto import runtime

MANIFEST = Path(__file__).resolve().parents[1] / "requirements-serverless-06.txt"
VERIFIED = {"pandas": "3.0.6", "numpy": "2.3.4", "scikit-learn": "1.7.2", "joblib": "1.5.2",
            "lightgbm": "4.7.0", "xgboost": "3.4.1", "optuna": "5.0.0", "mlflow": "3.18.0"}


@pytest.fixture
def simulated_environment(monkeypatch):
    versions = {**VERIFIED, "scipy": "provided", "databricks-sdk": "provided", "python-dotenv": "provided"}
    monkeypatch.setattr(runtime, "version", versions.__getitem__)
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(version_info=(3, 12, 3)))
    imports = []
    monkeypatch.setattr(runtime, "import_module", imports.append)
    return versions, imports


def test_serverless_pins_match_reported_versions(simulated_environment):
    result = runtime.diagnose_environment(MANIFEST, strict=True)
    assert result["expected_versions"] == VERIFIED
    assert result["python"] == "3.12.3" and not result["differences"]
    assert len(simulated_environment[1]) == len(runtime.IMPORTS)


@pytest.mark.parametrize("package", list(VERIFIED))
def test_any_serverless_version_drift_is_rejected_before_imports(simulated_environment, package):
    versions, imports = simulated_environment
    versions[package] = "different"
    with pytest.raises(RuntimeError, match="Entorno incompatible"):
        runtime.diagnose_environment(MANIFEST, strict=True)
    assert not imports


def test_local_preserves_other_versions_but_requires_pandas(simulated_environment):
    versions, _ = simulated_environment
    versions["numpy"] = "2.5.4"
    result = runtime.diagnose_environment(MANIFEST)
    assert result["differences"]["numpy"]["actual"] == "2.5.4"
    versions["pandas"] = "2.2.3"
    with pytest.raises(RuntimeError, match="pandas"):
        runtime.diagnose_environment(MANIFEST)


def test_serverless_python_version_is_verified(simulated_environment, monkeypatch):
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(version_info=(3, 12, 7)))
    with pytest.raises(RuntimeError, match="3.12.7"):
        runtime.diagnose_environment(MANIFEST, strict=True)
    assert runtime.diagnose_environment(MANIFEST)["python"] == "3.12.7"


def test_missing_dependency_and_native_import_failure_are_actionable(simulated_environment, monkeypatch):
    def missing(package):
        raise PackageNotFoundError(package)
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "version", missing)
        with pytest.raises(RuntimeError, match="Environment"):
            runtime.diagnose_environment(MANIFEST)
    def broken(module):
        raise ImportError("binary mismatch")
    monkeypatch.setattr(runtime, "import_module", broken)
    with pytest.raises(RuntimeError, match="no se puede importar"):
        runtime.diagnose_environment(MANIFEST)


def test_invalid_manifest_is_rejected(tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_text("pandas>=3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactas"):
        runtime.diagnose_environment(path)


def test_cached_pandas_from_before_install_requires_restart(simulated_environment, monkeypatch):
    monkeypatch.setattr(runtime, "import_module",
                        lambda name: SimpleNamespace(__version__="2.2.3") if name == "pandas" else None)
    with pytest.raises(RuntimeError, match="reinicia la sesión"):
        runtime.diagnose_environment(MANIFEST, strict=True)
