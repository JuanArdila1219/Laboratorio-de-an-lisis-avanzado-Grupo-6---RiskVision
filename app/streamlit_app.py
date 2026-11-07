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


# Secciones lógicas del formulario (orden visual); cualquier columna del esquema que no
# aparezca aquí se muestra al final en "Otros datos".
FORM_GROUPS = [
    ("👤 Perfil del cliente", ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]),
    ("⏰ Historial de atrasos (del más reciente al más antiguo)", ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]),
    ("🧾 Facturas de los últimos 6 meses (de la más reciente a la más antigua)", ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]),
    ("💵 Pagos de los últimos 6 meses (del más reciente al más antiguo)", ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]),
]

CUSTOM_CSS = """
<style>
.hero {
    background: linear-gradient(120deg, #0F4C81 0%, #1B6FBF 55%, #2E90D9 100%);
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
}
.hero h1 {color: #ffffff; font-size: 2.1rem; margin: 0 0 .4rem 0;}
.hero p {color: rgba(255, 255, 255, .93); font-size: 1rem; margin: 0;}
section[data-testid="stForm"] {
    border: 1px solid #d9e2ec;
    border-radius: 14px;
    padding: 1.2rem 1.4rem !important;
    background: #fbfdff;
}
.form-section-title {
    font-size: 1.02rem;
    font-weight: 600;
    color: #0F4C81;
    margin: .9rem 0 .2rem 0;
}
div[data-testid="stMetric"] {
    background: #f4f8fc;
    border: 1px solid #d9e2ec;
    border-radius: 12px;
    padding: .6rem .9rem;
}
div[data-testid="stExpander"] {border-radius: 12px;}
.verdict {border-radius: 14px; padding: 1rem 1.4rem; margin-bottom: 1rem;}
.verdict.low {background: linear-gradient(120deg, #1B7F4B 0%, #34A853 100%);}
.verdict.mid {background: linear-gradient(120deg, #B7791F 0%, #E6A817 100%);}
.verdict.high {background: linear-gradient(120deg, #B3261E 0%, #E5484D 100%);}
.verdict h3 {color: #ffffff; font-size: 1.25rem; margin: 0 0 .3rem 0;}
.verdict p {color: rgba(255, 255, 255, .95); margin: 0;}
.probarow {margin: .55rem 0;}
.probalabel {display: flex; justify-content: space-between; font-size: .92rem; color: #1F2937; margin-bottom: .3rem;}
.probtrack {background: #E8EEF5; border-radius: 999px; height: 14px; overflow: hidden;}
.probfill {height: 100%; border-radius: 999px;}
.probfill.paga {background: linear-gradient(90deg, #2E9E5B, #5BC17C);}
.probfill.impago {background: linear-gradient(90deg, #C0392B, #E5484D);}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
</style>
"""


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
    unknown = st.checkbox("No sé este dato", key=f"unk_{key}", help=label(schema, column))
    if unknown:
        return np.nan
    if is_int:
        return st.number_input(label(schema, column), min_value=int(low), max_value=int(high), value=int(round(default)), step=1, key=key)
    return st.number_input(label(schema, column), min_value=float(low), max_value=float(high), value=float(default), key=key)


def build_form(schema: dict) -> dict:
    """Dibuja el formulario agrupado en secciones lógicas y devuelve una fila de entrada."""
    row = {}
    columns = schema["numeric"] + schema["categorical"]
    groups = [(title, [c for c in cols if c in columns]) for title, cols in FORM_GROUPS]
    covered = [c for _, cols in groups for c in cols]
    rest = [c for c in columns if c not in covered]
    if rest:
        groups.append(("📋 Otros datos", rest))
    for title, cols in [g for g in groups if g[1]]:
        st.markdown(f'<div class="form-section-title">{title}</div>', unsafe_allow_html=True)
        ui = st.columns(3)
        for i, column in enumerate(cols):
            with ui[i % 3]:
                if column in schema["categorical"]:
                    options = schema["categories"][column] + [UNKNOWN]
                    choice = st.selectbox(label(schema, column), options, key=f"cat_{column}")
                    row[column] = np.nan if choice == UNKNOWN else choice
                else:
                    row[column] = numeric_field(schema, column, f"num_{column}")
    return row


def main() -> None:
    st.set_page_config(page_title="RiskVision", page_icon="🧭", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        st.title("Mi proyecto")
        st.error(
            "Falta `models/model.joblib` o `models/input_schema.json`. "
            "Ejecuta el notebook `notebooks/09_inferencia.ipynb` para generarlos."
        )
        st.stop()

    model, schema = get_model(), get_schema()
    app_title = schema.get("app_title", "RiskVision")
    st.markdown(
        f'<div class="hero"><h1>🧭 {app_title}</h1>'
        '<p>Herramienta de apoyo a la decisión. <b>No decide por sí sola.</b></p></div>',
        unsafe_allow_html=True,
    )

    with st.form("entrada"):
        row = build_form(schema)
        submitted = st.form_submit_button("Evaluar riesgo", type="primary", use_container_width=True)

    if not submitted:
        st.info("Completa los datos y pulsa **Evaluar**. Marca *No sé* en lo que no conozcas.")
        return

    try:
        result = predict_module.predict_with_scores(pd.DataFrame([row]), model, schema)
    except predict_module.InputValidationError as error:
        st.error(f"Revisa los datos: {error}")
        return

    def nombre_clase(c: object) -> str:
        """Traduce la etiqueta técnica de clase a lenguaje claro (binario: paga / impago)."""
        if len(class_names) == 2:
            return "No paga (impago)" if str(c) == str(class_names[1]) else "Sí paga (al día)"
        return f"Clase {c}"

    first = result.iloc[0]
    class_names = schema["class_names"]
    probas = {c: float(first[f"proba_{c}"]) for c in class_names}
    p_impago = probas.get(class_names[1]) if len(class_names) > 1 else None

    # Banner de veredicto: el color comunica el nivel de riesgo de un vistazo.
    if p_impago is None:
        nivel, emoji, level = "", "⚖️", "mid"
    elif p_impago < 0.30:
        nivel, emoji, level = "Riesgo bajo de impago", "🟢", "low"
    elif p_impago < 0.70:
        nivel, emoji, level = "Riesgo moderado de impago", "🟡", "mid"
    else:
        nivel, emoji, level = "Riesgo alto de impago", "🔴", "high"

    decision_txt = (
        "aprobar / seguir con el cliente"
        if str(first["decision"]) == str(class_names[0])
        else "denegar o revisar el crédito"
    )
    resumen = f"El modelo sugiere <b>{decision_txt}</b>."
    if p_impago is not None:
        resumen = f"Probabilidad de impago: <b>{p_impago:.1%}</b> · " + resumen
    st.markdown(
        f'<div class="verdict {level}"><h3>{emoji} {nivel}</h3><p>{resumen}</p></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Probabilidad por clase**")
        for c in class_names:
            p = probas[c]
            fill = "impago" if len(class_names) == 2 and str(c) == str(class_names[1]) else "paga"
            st.markdown(
                f'<div class="probarow">'
                f'<div class="probalabel"><span>{nombre_clase(c)}</span><span>{p:.1%}</span></div>'
                f'<div class="probtrack"><div class="probfill {fill}" style="width:{p * 100:.1f}%"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        detail = f"Clase más probable: **{nombre_clase(first['most_likely'])}**"
        if "expected_cost" in result.columns:
            detail += f" · costo esperado de esta decisión: **{first['expected_cost']:.2f}**"
        st.caption(detail)
    with right:
        if bool(first["needs_review"]):
            st.warning("**Revisión recomendada:** el modelo duda, o la decisión de menor costo difiere de la clase más probable.")
        else:
            st.success("El modelo está seguro de este resultado.")
        for message in result.attrs.get("warnings", []):
            st.warning(message)

    if "cost_matrix" in schema:
        with st.expander("¿Cómo se decide? (matriz de costos)"):
            st.markdown("La decisión es la de **menor costo esperado**, no necesariamente la más probable.")
            st.dataframe(pd.DataFrame(schema["cost_matrix"], index=[f"real {c}" for c in class_names], columns=[f"decide {c}" for c in class_names]))
    with st.expander("Uso responsable"):
        st.markdown(schema.get("responsible_use", DEFAULT_RESPONSIBLE_USE))


if __name__ == "__main__":  # `streamlit run` ejecuta el archivo como programa principal
    main()
