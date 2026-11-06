# Laboratorio-de-an-lisis-avanzado-Grupo-6---RiskVision
Caso de Negocio Default of Credit Card Clients

**App desplegada:** _enlace público de Streamlit (se completa en la clase 5)_

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
| `models/` | `model.joblib` y `ficha_modelo.md` |
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
