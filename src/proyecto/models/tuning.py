"""Búsqueda reproducible de LightGBM con Optuna (etapa 06 de RiskVision).

Recibe exclusivamente los predictores y etiquetas de entrenamiento EN MEMORIA,
o del subconjunto destinado a búsqueda. No carga archivos, no abre la prueba,
no compara contra una reserva ni reajusta automáticamente el mejor modelo.
El llamador debe garantizar el origen de los datos: no es posible identificar
una prueba entregada como tabla solo por sus valores.

Se mantiene PACKAGE_E, un Pipeline nuevo por pliegue y los mismos pliegues para
todos los ensayos. El mejor AUC de búsqueda tiene optimismo de selección; no
demuestra mejora. La referencia logística se construye sin ajustarla ni afinarla.
"""
from dataclasses import dataclass, fields
from numbers import Integral, Real

import numpy as np
import optuna
import pandas as pd
from lightgbm.basic import LightGBMError

from proyecto.config import RANDOM_STATE
from proyecto.data.cleaning import PREDICTORS
from proyecto.features.build import PACKAGE_E
from proyecto.models.candidates import make_lgbm, make_logistic
from proyecto.models.evaluate import repeated_folds, score_folds


@dataclass(frozen=True)
class SearchSpace:
    """Rangos inclusivos; tasas en escala logarítmica y lambda log si su mínimo > 0.

    Profundidades acotadas evitan árboles ilimitados. El número de hojas se
    restringe también a 2**max_depth en cada ensayo. reg_lambda puede empezar
    en cero: en ese caso se muestrea en escala lineal.
    """

    n_estimators: tuple[int, int] = (100, 600)
    learning_rate: tuple[float, float] = (0.01, 0.15)
    num_leaves: tuple[int, int] = (8, 63)
    max_depth: tuple[int, int] = (3, 10)
    min_child_samples: tuple[int, int] = (20, 150)
    subsample: tuple[float, float] = (0.6, 1.0)
    colsample_bytree: tuple[float, float] = (0.6, 1.0)
    reg_lambda: tuple[float, float] = (0.0001, 10.0)

    def __post_init__(self):
        integer_names = {"n_estimators", "num_leaves", "max_depth", "min_child_samples"}
        for field in fields(self):
            bounds = getattr(self, field.name)
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"{field.name}: se requiere una tupla (mínimo, máximo)")
            kind = Integral if field.name in integer_names else Real
            if any(not isinstance(v, kind) or isinstance(v, (bool, np.bool_)) for v in bounds):
                raise ValueError(f"{field.name}: límites de tipo numérico incorrecto")
            low, high = bounds
            if not all(np.isfinite(v) for v in bounds) or low > high:
                raise ValueError(f"{field.name}: rango no finito o invertido")
            if field.name == "reg_lambda":
                if low < 0:
                    raise ValueError("reg_lambda: debe ser no negativo")
            elif low <= 0:
                raise ValueError(f"{field.name}: debe ser positivo")
            if field.name in {"learning_rate", "subsample", "colsample_bytree"} and high > 1:
                raise ValueError(f"{field.name}: el máximo no puede superar 1")
        if self.num_leaves[0] < 2:
            raise ValueError("num_leaves: se requieren al menos 2 hojas")
        if self.num_leaves[0] > 2 ** self.max_depth[0]:
            raise ValueError("num_leaves: el mínimo no cabe en la profundidad mínima")


@dataclass(frozen=True)
class TuningConfig:
    """Presupuesto por llamada y semillas; Optuna siempre usa ensayos secuenciales.

    Con datos, orden, configuración, versiones y plataforma iguales, se reutilizan
    particiones y secuencia de propuestas. No promete igualdad entre plataformas.
    """

    n_trials: int = 12
    seed: int = RANDOM_STATE
    n_splits: int = 5
    n_repeats: int = 3
    model_n_jobs: int = 1

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            minimum = 0 if field.name == "seed" else (2 if field.name == "n_splits" else 1)
            if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < minimum:
                raise ValueError(f"{field.name}: debe ser un entero >= {minimum}")
            object.__setattr__(self, field.name, int(value))
        if self.seed > 2**32 - 1:
            raise ValueError("seed: fuera del rango de semillas admitido")


class TrialEvaluationError(RuntimeError):
    """Un ensayo no pudo evaluarse; Optuna lo registra FAIL y continúa."""


class MlflowTrialLogger:
    """Callback secuencial, sin configuración de conexión ni reanudación.

    Un objeto por búsqueda. Las llamadas repetidas para un ensayo no crean otro
    run. Un error de seguimiento detiene la búsqueda: no se oculta ni se reintenta
    creando duplicados. No garantiza registro tras terminar a la fuerza el proceso.
    """

    def __init__(self, experiment_id, *, params=None, tags=None, client=None):
        if client is None:
            from mlflow import MlflowClient
            client = MlflowClient()
        self.client = client
        self.experiment_id = str(experiment_id)
        self.params = dict(params or {})
        self.tags = dict(tags or {})
        self.run_ids = {}
        self._study_name = None
        self._errors = {}

    def __call__(self, study, trial):
        if not trial.state.is_finished():
            return
        if self._study_name not in (None, study.study_name):
            raise ValueError("Usa un MlflowTrialLogger nuevo por búsqueda")
        self._study_name = study.study_name
        if trial.number in self._errors:
            raise RuntimeError("El seguimiento de este ensayo falló; revisar su run antes de continuar") from self._errors[trial.number]
        if trial.number in self.run_ids:
            return
        interrupted = trial.user_attrs.get("interrupted", False)
        status = ("KILLED" if interrupted or trial.state == optuna.trial.TrialState.PRUNED else
                  "FINISHED" if trial.state == optuna.trial.TrialState.COMPLETE else "FAILED")
        tags = {**self.tags, "role": "optuna_trial", "optuna_state": trial.state.name,
                "trial_number": str(trial.number), "optuna_study": study.study_name,
                "outcome": "interrupted" if interrupted else trial.state.name.lower(),
                "failure_type": trial.user_attrs.get("failure_type", ""),
                "failure_message": trial.user_attrs.get("failure_message", "")}
        try:
            run = self.client.create_run(self.experiment_id, tags=tags, run_name=f"trial-{trial.number:03d}")
        except BaseException as error:
            self._errors[trial.number] = error
            raise
        run_id = run.info.run_id
        self.run_ids[trial.number] = run_id
        try:
            for name, value in {**self.params, **trial.params, "trial_number": trial.number}.items():
                self.client.log_param(run_id, name, "None" if value is None else value)
            if trial.duration is not None:
                self.client.log_metric(run_id, "duration_seconds", trial.duration.total_seconds())
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
                self.client.log_metric(run_id, "roc_auc_cv", float(trial.value))
            for name in ("roc_auc_std", "avg_precision_mean", "penalty_mean"):
                value = trial.user_attrs.get(name)
                if value is not None and np.isfinite(value):
                    self.client.log_metric(run_id, name, float(value))
            self.client.set_tag(run_id, "mlflow_run_id", run_id)
            self.client.set_terminated(run_id, status=status)
        except BaseException as error:
            self._errors[trial.number] = error
            self.client.set_terminated(run_id, status="FAILED")
            raise


def suggest_parameters(trial, space: SearchSpace | None = None) -> dict:
    """Propone los ocho hiperparámetros con restricciones compatibles entre sí."""
    space = space or SearchSpace()
    depth = trial.suggest_int("max_depth", *space.max_depth)
    leaves_high = min(space.num_leaves[1], 2**depth)
    return {
        "max_depth": depth,
        "num_leaves": trial.suggest_int("num_leaves", space.num_leaves[0], leaves_high),
        "n_estimators": trial.suggest_int("n_estimators", *space.n_estimators),
        "learning_rate": trial.suggest_float("learning_rate", *space.learning_rate, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", *space.min_child_samples),
        "subsample": trial.suggest_float("subsample", *space.subsample),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *space.colsample_bytree),
        "reg_lambda": trial.suggest_float("reg_lambda", *space.reg_lambda, log=space.reg_lambda[0] > 0),
    }


def make_trial_pipeline(params: dict, *, seed: int = RANDOM_STATE, model_n_jobs: int = 1):
    """Construye un Pipeline independiente mediante la fábrica existente.

    No acepta parámetros de variables, clases o preprocesamiento. Se mantiene
    subsample_freq=1 de candidates.py para que subsample tenga efecto.
    """
    names = {field.name for field in fields(SearchSpace)}
    if set(params) != names:
        raise ValueError("params debe contener exactamente los ocho hiperparámetros de SearchSpace")
    SearchSpace(**{name: (value, value) for name, value in params.items()})
    config = TuningConfig(seed=seed, model_n_jobs=model_n_jobs)
    pipe = make_lgbm(features=PACKAGE_E)
    pipe.set_params(**{f"model__{name}": value for name, value in params.items()},
                    model__random_state=config.seed, model__n_jobs=config.model_n_jobs,
                    model__deterministic=True, model__force_col_wise=True)
    return pipe


def make_reference_pipeline():
    """Logística con E y parámetros originales, sin ajustar ni optimizar."""
    return make_logistic(features=PACKAGE_E)


def make_search_folds(X: pd.DataFrame, y, config: TuningConfig | None = None):
    """Valida entradas y construye una sola lista reproducible de índices.

    X contiene las 23 columnas originales ya limpias, sin ID, objetivo ni
    variables precalculadas. Si y es Series, su índice debe coincidir con X.
    La unicidad de clientes debe comprobarse antes de retirar el ID (make_xy).
    """
    config = config or TuningConfig()
    if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates or set(X.columns) != set(PREDICTORS):
        raise ValueError("X debe contener exactamente los 23 predictores originales, sin ID ni objetivo")
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("Los índices de X e y no coinciden")
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(X) or not len(labels):
        raise ValueError("X e y deben tener filas compatibles y no estar vacíos")
    if not pd.Series(labels).isin([0, 1]).all() or set(labels) != {0, 1}:
        raise ValueError("y debe contener ambas clases 0 y 1, sin valores faltantes")
    if min(int((labels == value).sum()) for value in (0, 1)) < config.n_splits:
        raise ValueError("Cada clase debe tener al menos n_splits observaciones")
    folds = repeated_folds(X, labels, n_splits=config.n_splits, n_repeats=config.n_repeats, seed=config.seed)
    for train_idx, valid_idx in folds:
        train_idx.flags.writeable = valid_idx.flags.writeable = False
    return tuple(folds)


def create_objective(X_train: pd.DataFrame, y_train, *, config: TuningConfig | None = None,
                     space: SearchSpace | None = None):
    """Crea el objetivo ROC AUC y fija copias de datos y pliegues antes de buscar.

    Se capturan errores de datos/ejecución esperables por ensayo, no errores de
    programación como KeyError/TypeError ni interrupciones del usuario.
    """
    config, space = config or TuningConfig(), space or SearchSpace()
    folds = make_search_folds(X_train, y_train, config)
    X, y = X_train.copy(deep=True), np.asarray(y_train).copy()

    def objective(trial):
        params = suggest_parameters(trial, space)
        try:
            scores = score_folds(lambda: make_trial_pipeline(params, seed=config.seed,
                                                            model_n_jobs=config.model_n_jobs), X, y, folds)
            auc = scores["roc_auc"].to_numpy(dtype=float)
            if len(auc) != len(folds) or not np.isfinite(auc).all() or ((auc < 0) | (auc > 1)).any():
                raise ValueError("La evaluación no devolvió un AUC válido para cada pliegue")
        except (ValueError, RuntimeError, FloatingPointError, LightGBMError) as error:
            trial.set_user_attr("failure_type", type(error).__name__)
            trial.set_user_attr("failure_message", str(error)[:500])
            raise TrialEvaluationError(f"No se pudo evaluar el ensayo: {error}") from error
        trial.set_user_attr("roc_auc_per_fold", auc.tolist())
        trial.set_user_attr("roc_auc_std", float(auc.std(ddof=1)))
        trial.set_user_attr("avg_precision_mean", float(scores["avg_precision"].mean()))
        trial.set_user_attr("penalty_mean", float(scores["penalty"].mean()))
        return float(auc.mean())

    return objective


def run_search(X_train: pd.DataFrame, y_train, *, config: TuningConfig | None = None,
               space: SearchSpace | None = None, trial_logger=None) -> optuna.study.Study:
    """Ejecuta el presupuesto explícito, sin persistencia, refit ni comparación final.

    Los fallos esperables cuentan en el presupuesto y quedan como FAIL. Si todos
    fallan, se devuelve el estudio para diagnóstico, sin inventar parámetros:
    best_trial/best_value no existen en ese caso y Optuna lanza ValueError.
    trial_logger es un callback opcional; sin él no se importa ni usa MLflow.
    No configura MLflow, no carga joblib y no toca test.joblib. Los resultados
    solo describen validación de búsqueda, nunca una mejora ya demostrada.
    """
    config, space = config or TuningConfig(), space or SearchSpace()
    objective = create_objective(X_train, y_train, config=config, space=space)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed),
                                pruner=optuna.pruners.NopPruner())
    study.set_user_attr("features", list(PACKAGE_E))
    study.set_user_attr("config", {field.name: getattr(config, field.name) for field in fields(config)})
    study.set_user_attr("search_space", {field.name: list(getattr(space, field.name)) for field in fields(space)})
    study.set_user_attr("metric", "roc_auc")
    study.set_user_attr("selection_warning", "Mejor resultado de búsqueda; puede tener optimismo de selección")
    if trial_logger is None:
        study.optimize(objective, n_trials=config.n_trials, n_jobs=1, catch=(TrialEvaluationError,), show_progress_bar=False)
    else:
        def tracked_objective(trial):
            try:
                return objective(trial)
            except BaseException as error:
                trial.set_user_attr("interrupted", isinstance(error, (KeyboardInterrupt, SystemExit)))
                if "failure_type" not in trial.user_attrs:
                    trial.set_user_attr("failure_type", type(error).__name__)
                    trial.set_user_attr("failure_message", str(error)[:500])
                raise

        try:
            study.optimize(tracked_objective, n_trials=config.n_trials, n_jobs=1,
                           catch=(TrialEvaluationError,), callbacks=[trial_logger], show_progress_bar=False)
        except (KeyboardInterrupt, SystemExit):
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.RUNNING and trial.user_attrs.get("interrupted"):
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            raise
        finally:
            # Optuna puede omitir callbacks cuando propaga una interrupción/error.
            # El logger es idempotente; completa el registro sin repetir ensayos.
            for trial in study.trials:
                if trial.state.is_finished():
                    trial_logger(study, trial)
    return study
