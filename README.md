# Laboratorio-de-an-lisis-avanzado-Grupo-6---RiskVision
Caso de Negocio Default of Credit Card Clients

**App desplegada:** _enlace público pendiente de publicar en Streamlit Community Cloud_ (en local: `streamlit run app/streamlit_app.py`).

**Resultado:** LightGBM con 5 variables de comportamiento de pago, ROC AUC 0,787 en validación cruzada y 0,782 en la prueba (ya expuesta, no independiente). La app marca como riesgo a quien tenga probabilidad de incumplir ≥ 0,20. Detalle y límites: `models/ficha_modelo.md` e `reports/informe_tecnico.md`.

## Qué contiene
Estructura obligatoria del curso. Cada carpeta tiene un propósito:

| Carpeta | Contenido |
|---|---|
| `.specify/` y `.claude/` | Entorno de IA: constitución del proyecto, Spec Kit (`/speckit-…`) y subagentes |
| `specs/` | Lo que genera Spec Kit: `spec.md`, `plan.md`, `tasks.md` de cada pieza (aparece al usarlo) |
| `especificaciones/` | Guías que se llenan **antes** de programar cada etapa |
| `data/raw/` | Datos originales, intocables |
| `data/processed/` | Datos limpios generados por el código |
| `notebooks/` | Un notebook por etapa; solo orquestan |
| `src/proyecto/` | Código reutilizable (data, features, models, tracking, inference) |
| `tests/` | Pruebas con `pytest` |
| `models/` | `model.joblib`, `input_schema.json` (lo que carga la app) y `ficha_modelo.md`; `_local/` (no se sube) guarda el paquete verificable |
| `app/` | `streamlit_app.py` |
| `reports/` | Una sección por hito e `informe_tecnico.md` |
| `presentacion/` | `ejecutiva.md` (máximo 5 diapositivas) |

## Cómo ejecutarlo
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # en Mac/Linux: .venv/bin/python
cp .env.example .env     # completa PROFILE o DATABRICKS_HOST y DATABRICKS_TOKEN
python scripts/check_databricks.py
pytest
streamlit run app/streamlit_app.py
```

Para repetir la validación, el registro y la exportación del modelo (07→09; **no abre la prueba**): `python scripts/run_etapas_07_09.py --approved-by "Nombre" --out resultados.json [--approve-definitive --register]`. El resumen de resultados está en `notebooks/10_ejecucion_real_07_09.ipynb`. El registro se hace en Databricks si `.env` tiene conexión y, si no, en MLflow local.

## De dónde viene este repositorio
Nació de la rama `feature/estudiantes` del repositorio del curso (ver `LEEME-PRIMERO.md`): una persona del grupo la clonó con
`git clone --branch feature/estudiantes --single-branch …`, la apuntó a este repo con `git remote set-url origin …` y hizo `push`. Los demás integrantes
clonan **este** repositorio. Para arrancar, sigue `material-del-curso/guias/montaje-proyecto.md`.

## Datos
Dataset *Default of Credit Card Clients* (30.000 clientes de un banco de Taiwán), licencia **CC BY 4.0**.
Cita: Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
El archivo `.xls` va en `data/raw/` (esa carpeta no se sube a GitHub). Los datos procesados de `data/processed/` (`train.joblib` y `test.joblib`, la partición 80/20 con semilla 42) **sí se suben**, para que todos usen la misma; también se regeneran con `notebooks/03_preprocesamiento.ipynb`. El conjunto de prueba (`test.joblib`) no se usa para decidir nada hasta la validación final.

## Equipo
| Integrante | Rol |
|---|---|
| Juan David Ardila Arboleda | Analista de Datos |
| Lina Isabel Vasquez Villegas | Analista de Datos |
| Andres Felipe Orozco Roa | Analista de Datos |
| Cristina Garces | Analista de Datos |
| Estefania Cervera | Analista de Datos |

Especificación del proyecto: [`especificacion-proyecto.md`](especificacion-proyecto.md) · Especificaciones por etapa: [`especificaciones/`](especificaciones/)
