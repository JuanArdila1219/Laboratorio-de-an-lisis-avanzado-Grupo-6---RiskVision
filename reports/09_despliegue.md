# Despliegue de la app

> **Estado:** la app corre y pasa las comprobaciones en local; **el despliegue en Streamlit Community Cloud sigue pendiente** (requiere la cuenta del grupo).
> Lo que falta lo completa quien despliegue: enlace, prueba en ventana privada y ensayo cronometrado.
> Guía paso a paso: `material-del-curso/guias/streamlit-despliegue.md` y `material-del-curso/kit/guia-despliegue-y-presentacion.md`.

## 1. Enlace y estado

| Dato | Valor |
|---|---|
| Enlace público de la app | |
| Plataforma | Streamlit Community Cloud |
| Repo, rama y archivo principal | |
| Versión de Python elegida | ☐ 3.12  ☐ otra → motivo: _______________ |
| ¿Se comprobó en una **ventana privada**? | ☐ sí  ☐ no |
| ¿Se desplegó? | ☐ sí  ☐ no → **plan B** (apartado 4) |

## 2. Decisiones de despliegue y su motivo

| Decisión | Motivo |
|---|---|
| La app carga `models/model.joblib` desde el repo y **no consulta Databricks** | No poner credenciales en la nube y no depender de la cuota de ninguna plataforma |
| Versiones fijadas en `requirements.txt` | El modelo es un pipeline serializado: necesita las mismas versiones (Python 3.12, scikit-learn, lightgbm) |
| Qué NO se subió al repo (y por qué) | `.env`, `mlflow_local.db` y `models/_local/` (traspaso y paquete con huellas): credenciales y archivos pesados o locales |
| | |

## 3. Pruebas del enlace público

| Caso | Datos usados | Resultado esperado | Resultado real |
|---|---|---|---|
| Normal | Cliente al día (`PAY_*` ≤ 0) | Resultado «0» | Comprobado en local con `streamlit.testing`: «0» |
| Límite (cambia la decisión) | | | |
| Inválido (debe rechazarse con un mensaje claro) | Edad 5 | Rechazo con mensaje | `AGE: debe estar entre 18 y 100` (comprobado en `scripts/run_etapas_07_09.py` y `tests/`) |
| Con un campo en «No sé» | | | |

## 4. Problemas encontrados y plan B

- Errores de despliegue que aparecieron y cómo se resolvieron (qué decía el registro y qué se cambió): _______________
- **Plan B** (si la app falla en vivo): captura con fecha y repo → `_______________`; app en local → `_______________`
- Si **no se desplegó**: **plan de despliegue escrito** (qué se intentó, qué error dio, qué harían): _______________

## 5. Ensayo cronometrado (clase 5)

| Parte | Tiempo previsto | Tiempo real |
|---|---|---|
| Caso de negocio | 2 min | |
| App (demo en vivo) | 4 min | |
| Resultados | 3 min | |
| Cierre | 1 min | |
| **Total** | **10 min** | |

- ¿Se pasó de 10 minutos? ☐ no  ☐ sí → qué se recortó: _______________
- Retroalimentación recibida (lo mejor y la corrección que propusieron): _______________
- **Corrección concreta que hicimos por ella:** _______________
- Quién habla de qué y quién responde las preguntas técnicas: _______________
