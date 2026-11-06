"""App de tu proyecto: un formulario que se construye solo a partir de `models/input_schema.json`.

- Carga `models/model.joblib` desde ESTE repositorio. NO consulta Databricks: no se ponen credenciales
  en la nube, y la app no depende de la cuota de ninguna plataforma.
- Valida las entradas: si un dato es imposible, explica qué corregir en vez de responder cualquier cosa.

Ejecutar en local (desde la raíz de tu repo):  streamlit run app/streamlit_app.py

Para personalizarla (título, etiquetas, aviso de uso responsable) edita `models/input_schema.json`
(claves `app_title`, `labels`, `responsible_use`) o este archivo. Pídele a Claude cambios pequeños,
de uno en uno, y pruébalos con `streamlit run`.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
SCHEMA_PATH = ROOT / "models" / "input_schema.json"
UNKNOWN = "(no sé)"

# El paquete de src/ puede llamarse `proyecto` o el nombre que le hayas puesto: lo buscamos.
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
PACKAGE = next((p.name for p in sorted(SRC.iterdir()) if p.is_dir() and (p / "__init__.py").exists()), "proyecto")
predict_module = importlib.import_module(f"{PACKAGE}.inference.predict")
schema_module = importlib.import_module(f"{PACKAGE}.inference.schema")

DEFAULT_RESPONSIBLE_USE = (
    "- Esta herramienta **apoya** una decisión; la decisión final es de una persona.\n"
    "- Fue entrenada con los datos de este proyecto: no la uses fuera de ese contexto sin validarla de nuevo.\n"
    "- Revisa los **límites del modelo** en `models/ficha_modelo.md` antes de usarla."
)


@st.cache_resource
def get_model():
    return predict_module.load_model(MODEL_PATH)


@st.cache_data
def get_schema() -> dict:
    return schema_module.load_schema(SCHEMA_PATH)


def label(schema: dict, column: str) -> str:
    return schema.get("labels", {}).get(column, column)


def numeric_field(schema: dict, column: str, key: str):
    """Un campo numérico con su rango y mediana, y una casilla 'no sé' que deja el dato vacío."""
    low, high = schema.get("valid_range", {}).get(column) or schema["observed_range"][column]
    default = float(schema["median"][column])
    is_int = column in schema.get("integer_columns", [])
    if high <= low:
        high = low + 1
    default = min(max(default, low), high)
    unknown = st.checkbox(f"No sé: {label(schema, column)}", key=f"unk_{key}")
    if unknown:
        return np.nan
    if is_int:
        return st.number_input(label(schema, column), min_value=int(low), max_value=int(high), value=int(round(default)), step=1, key=key)
    return st.number_input(label(schema, column), min_value=float(low), max_value=float(high), value=float(default), key=key)


def build_form(schema: dict) -> dict:
    """Dibuja el formulario (tres columnas) y devuelve una fila de entrada."""
    row = {}
    columns = schema["numeric"] + schema["categorical"]
    cols = st.columns(3)
    for i, column in enumerate(columns):
        with cols[i % 3]:
            if column in schema["categorical"]:
                options = schema["categories"][column] + [UNKNOWN]
                choice = st.selectbox(label(schema, column), options, key=f"cat_{column}")
                row[column] = np.nan if choice == UNKNOWN else choice
            else:
                row[column] = numeric_field(schema, column, f"num_{column}")
    return row


def main() -> None:
    st.set_page_config(page_title="Mi proyecto", page_icon="🧭", layout="wide")

    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        st.title("Mi proyecto")
        st.error(
            "Falta `models/model.joblib` o `models/input_schema.json`. "
            "Ejecuta el notebook `notebooks/09_inferencia.ipynb` para generarlos."
        )
        st.stop()

    model, schema = get_model(), get_schema()
    st.title(schema.get("app_title", "Mi proyecto"))
    st.caption("Herramienta de apoyo a la decisión. **No decide por sí sola.**")

    with st.form("entrada"):
        row = build_form(schema)
        submitted = st.form_submit_button("Evaluar", type="primary")

    if not submitted:
        st.info("Completa los datos y pulsa **Evaluar**. Marca *No sé* en lo que no conozcas.")
        return

    try:
        result = predict_module.predict_with_scores(pd.DataFrame([row]), model, schema)
    except predict_module.InputValidationError as error:
        st.error(f"Revisa los datos: {error}")
        return

    first = result.iloc[0]
    class_names = schema["class_names"]
    left, right = st.columns(2)
    with left:
        st.metric("Resultado sugerido", str(first["decision"]))
        detail = f"Clase más probable: **{first['most_likely']}**"
        if "expected_cost" in result.columns:
            detail += f" · costo esperado de esta decisión: **{first['expected_cost']:.2f}**"
        st.caption(detail)
        if bool(first["needs_review"]):
            st.warning("**Revisión recomendada:** el modelo duda, o la decisión de menor costo difiere de la clase más probable.")
        else:
            st.success("El modelo está seguro de este resultado.")
        for message in result.attrs.get("warnings", []):
            st.warning(message)
    with right:
        st.write("**Probabilidad por clase**")
        st.bar_chart(pd.DataFrame({"probabilidad": [first[f"proba_{c}"] for c in class_names]}, index=class_names))

    if schema.get("decision_threshold") is not None:
        with st.expander("¿Cómo se decide? (umbral de trabajo)"):
            st.markdown(
                f"Se marca al cliente como **riesgo de incumplir** si su probabilidad es **{schema['decision_threshold']:.0%} o más**. "
                "Es un umbral bajo a propósito: detectar a quien incumple se valora más que una falsa alarma "
                "(hipótesis de 5 a 1 del grupo, **no un costo medido del banco**). "
                "Por eso se marca a alrededor de un tercio de los clientes; no significa que todos vayan a incumplir. "
                "La decisión final es de una persona."
            )
    if "cost_matrix" in schema:
        with st.expander("¿Cómo se decide? (matriz de costos)"):
            st.markdown("La decisión es la de **menor costo esperado**, no necesariamente la más probable.")
            st.dataframe(pd.DataFrame(schema["cost_matrix"], index=[f"real {c}" for c in class_names], columns=[f"decide {c}" for c in class_names]))
    with st.expander("Uso responsable"):
        st.markdown(schema.get("responsible_use", DEFAULT_RESPONSIBLE_USE))


if __name__ == "__main__":  # `streamlit run` ejecuta el archivo como programa principal
    main()
