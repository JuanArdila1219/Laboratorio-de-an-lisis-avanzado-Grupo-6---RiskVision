"""Diagnóstico sin instalaciones para la sesión de la etapa 06."""
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


IMPORTS = {"pandas": "pandas", "numpy": "numpy", "scikit-learn": "sklearn",
           "joblib": "joblib", "lightgbm": "lightgbm", "xgboost": "xgboost",
           "optuna": "optuna", "mlflow": "mlflow", "scipy": "scipy",
           "databricks-sdk": "databricks.sdk", "python-dotenv": "dotenv"}


def diagnose_environment(requirements_path, *, strict=False):
    """Comprueba versiones e imports antes de deserializar datos.

    pandas se exige también en local por la compatibilidad del artefacto.
    strict exige las versiones verificadas de Serverless y Python 3.12.3.
    No modifica dependencias ni métodos de pandas. Las librerías auxiliares
    sin versión verificada se comprueban y registran sin inventar restricciones.
    """
    pins = {}
    for line in Path(requirements_path).read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if requirement:
            parts = requirement.split("==")
            if len(parts) != 2 or not all(parts) or parts[0] in pins:
                raise ValueError("El manifiesto requiere versiones exactas y nombres únicos")
            pins[parts[0]] = parts[1]
    if set(pins) != set(IMPORTS) - {"scipy", "databricks-sdk", "python-dotenv"}:
        raise ValueError("El manifiesto no coincide con las dependencias de la etapa 06")
    effective = {}
    for package in IMPORTS:
        try:
            effective[package] = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Falta {package}; configura Environment y reinicia la sesión") from error
    python_version = ".".join(str(value) for value in sys.version_info[:3])
    differences = {name: {"expected": expected, "actual": effective[name]}
                   for name, expected in pins.items() if effective[name] != expected}
    if effective["pandas"] != "3.0.6" or (strict and (differences or python_version != "3.12.3")):
        raise RuntimeError(f"Entorno incompatible: Python={python_version}; diferencias={differences}. "
                           "Configura requirements-serverless-06.txt antes de cargar train.joblib "
                           "y reinicia la sesión; no aplicar parches a pandas")
    for package, module in IMPORTS.items():
        try:
            imported = import_module(module)
        except Exception as error:
            raise RuntimeError(f"{package} está instalado pero no se puede importar; revisa Environment") from error
        if package in pins and getattr(imported, "__version__", effective[package]) != effective[package]:
            raise RuntimeError(f"{package} cargado en memoria difiere del instalado; reinicia la sesión "
                               "y ejecuta desde la primera celda")
    return {"mode": "serverless" if strict else "local", "python": python_version,
            "versions": effective, "expected_versions": pins, "differences": differences,
            "auxiliary_versions_not_pinned": ["scipy", "databricks-sdk", "python-dotenv"]}
