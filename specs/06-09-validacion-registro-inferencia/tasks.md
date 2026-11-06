# Tareas (retroactivas) · Etapas 06–09

Marca `[x]` solo si su prueba o comprobación existe y pasa.

- [x] T1 Adoptar la configuración de Estefanía y documentarla — `reports/06_adopcion_estefania.md`, `tests/test_decision_handoff.py`
- [x] T2 Validación interna (OOF + reserva), calibración y umbral — `tests/test_validation.py`
- [x] T3 Contrato 07→08 firmado y verificado por huella — `tests/test_decision_handoff.py`
- [x] T4 Ajuste final y paquete verificable (ida y vuelta idéntica) — `tests/test_persistence.py`, `tests/test_pipeline_serialization.py`
- [x] T5 Registro con versión y alias (MLflow **local**) — `reports/registro_08.json`, `tests/test_model_registry.py`
- [ ] T6 Registro en **Databricks** — requiere `.env` con `PROFILE` o `DATABRICKS_HOST`/`DATABRICKS_TOKEN`; repetir con `scripts/run_etapas_07_09.py --approve-definitive --register`
- [x] T7 Exportar `models/model.joblib` y `models/input_schema.json` — `tests/test_modelo_exportado.py`, `tests/test_final_run.py`
- [x] T8 Inferencia con validación de entradas y umbral explícito — `tests/test_inference.py`, `tests/test_inference_service.py`
- [x] T9 App local que carga el modelo (verificada con `streamlit.testing`) — `app/streamlit_app.py`
- [x] T10 Ficha del modelo, informe del umbral e informe técnico con cifras reales — `models/ficha_modelo.md`, `reports/`
- [ ] T11 **Aprobación definitiva por escrito del grupo** (hoy la registró el usuario que ejecutó el flujo)
- [ ] T12 Confirmar con el instructor el costo real y la capacidad de revisión y recalcular el umbral
- [ ] T13 Despliegue público en Streamlit Community Cloud, prueba en ventana privada y ensayo de 10 minutos — `reports/09_despliegue.md`
