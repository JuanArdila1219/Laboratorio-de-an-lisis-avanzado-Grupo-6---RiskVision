"""Constructor del Pipeline: variables + preprocesamiento + modelo.

Todo lo que se aprende de los datos (mediana, escala, categorías) vive dentro del Pipeline, así que
al validar o predecir se ajusta solo con las filas de entrenamiento de ese momento. El Pipeline recibe
las filas ya limpias (`clean`) sin `ID` ni objetivo (`make_xy`) y calcula él mismo las variables nuevas,
de modo que la app y la inferencia puedan pasarle los datos de un cliente tal cual.
"""
from functools import partial

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from proyecto.config import CATEGORICAL
from proyecto.data.cleaning import PAY_COLS
from proyecto.features.build import build_features


def add_features_and_drop(df, names=(), drop=()):
    """Añade las variables pedidas y quita las columnas indicadas (para medir el modelo sin `SEX` o sin `PAY_0`)."""
    return build_features(df, names).drop(columns=list(drop))


def numeric_columns(X, excluded=()):
    """Selecciona en su orden las columnas no categóricas, sin aprender datos.

    Se define en el módulo para que Joblib pueda importarla al recuperar el
    pipeline. Las exclusiones se fijan con partial al construir cada pipeline.
    """
    return [c for c in X.columns if c not in excluded]


def build_pipeline(model, features=(), drop=(), linear=False, pay_as_category=False):
    """Pipeline completo.

    features: nombres de `proyecto.features.build.FEATURES` que se añaden.
    drop: columnas que se quitan antes de modelar.
    linear: estandariza los números (necesario para la regresión logística, innecesario para árboles).
    pay_as_category: trata PAY_* como estados (one-hot) y no como cantidades; útil en la logística.
    """
    cat_cols = [c for c in [*CATEGORICAL, *(PAY_COLS if pay_as_category else [])] if c not in drop]

    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if linear:
        numeric_steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([("cat", categorical, cat_cols),
                              ("num", Pipeline(numeric_steps), partial(numeric_columns, excluded=tuple(cat_cols)))])
    return Pipeline([
        ("features", FunctionTransformer(add_features_and_drop, kw_args={"names": tuple(features), "drop": tuple(drop)})),
        ("prep", prep),
        ("model", model),
    ])
