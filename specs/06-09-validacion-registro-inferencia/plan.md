# Plan (retroactivo) · Etapas 06–09: optimización, validación, registro e inferencia

> **Retroactivo:** este plan se escribió *después* de implementar las etapas, para dejar constancia de lo hecho y de lo pendiente.
> Lo que pide CLAUDE.md (plan y tareas aprobados antes de programar) no se cumplió para estas etapas; queda como lección para la siguiente pieza.

## Objetivo
Dejar un modelo validado, persistido, registrado con versión y alias, y servido por una app local, con decisiones congeladas antes de abrir la prueba.

## Decisiones (con su motivo)
| Decisión | Motivo |
|---|---|
| Se adopta el LightGBM de Estefanía (paquete E) sin repetir Optuna | Su mejora optimizada cruzó el cero; los parámetros de partida son más simples (`reports/06_adopcion_estefania.md`) |
| Calibración: ninguna | No bajó Brier ni log loss en validación cruzada |
| Evaluación: OOF 5 pliegues + reserva interna 25 % + prueba abierta una vez | La prueba solo confirma; no decide. Ya estaba expuesta: se declara |
| Umbral de trabajo 0,20 (validación 1/6) | La penalización 5:1 es casi plana cerca del óptimo y el 0,20 reduce 20 % la carga de revisión (`reports/07_umbral_y_resultados.md`) |
| Ajuste final solo en 08 con los 24.000 clientes | Un único responsable del ajuste final evita fuga entre etapas |
| Registro en MLflow local | No hay conexión con Databricks en esta máquina; el curso permite el respaldo local |
| La app carga `models/model.joblib` y no consulta Databricks | Regla 8 de CLAUDE.md |

## Piezas
`src/proyecto/models/{tuning,validation,decision,persistence,final_run}.py`, `src/proyecto/tracking/{reporting,model_registry}.py`,
`src/proyecto/inference/{predict,service}.py`, `scripts/run_etapas_07_09.py`, `notebooks/06`–`10`, `app/streamlit_app.py`.

## Riesgos
Prueba expuesta; costo 5 a 1 hipotético; datos de 2005 de un solo banco; segmento 45–60 años con menor AUC; aprobación definitiva dada por el usuario en nombre del grupo.
