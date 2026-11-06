# Informe técnico — RiskVision (Grupo 6)

> Ensambla las secciones de `reports/`. **Cada decisión va con su motivo.** Cifras reproducibles con `scripts/run_etapas_07_09.py` y los notebooks.

**Grupo:** Juan David Ardila Arboleda, Lina Isabel Vasquez Villegas, Andres Felipe Orozco Roa, Cristina Garces y Estefania Cervera (analistas de datos).
**Repositorio:** https://github.com/aforozco1808-dotcom/Laboratorio-de-an-lisis-avanzado-Grupo-6---RiskVision
**App:** _enlace público pendiente de despliegue en Streamlit Community Cloud (ver `reports/09_despliegue.md`); la app corre en local con `streamlit run app/streamlit_app.py`._

## Resumen ejecutivo

El área de riesgo de crédito necesita decidir a qué clientes de tarjeta hacer seguimiento preventivo antes de que dejen de pagar. Entrenamos un modelo
(LightGBM con cinco variables de comportamiento de pago) con 24.000 clientes de un banco de Taiwán. Ordena a los clientes por riesgo con un **ROC AUC de
0,787 en validación cruzada y 0,782 en la prueba** (el azar vale 0,50). Con el punto de trabajo elegido, la app marca al **34 %** de los clientes y
detecta a **68 %** de quienes incumplirán. **Límites principales:** datos de un solo banco de 2005; la relación de costo 5 a 1 y el umbral son hipótesis del
grupo, no costos medidos; la prueba ya estaba expuesta, así que no es una evaluación independiente. El modelo apoya una decisión humana y no debe usarse
para negar crédito automáticamente.

## 1. Caso de negocio

> Fuente: `reports/00_caso.md`.

- **Quién decide y qué:** el área de riesgo de crédito decide si ajusta el límite, hace seguimiento preventivo u ofrece reestructurar la deuda.
- **Pregunta:** ¿qué clientes tienen alta probabilidad de no pagar el próximo mes?
- **Costo de equivocarse:** hipótesis del grupo de que no detectar a quien incumple (falso negativo) pesa 5 veces una falsa alarma (`[[0,1],[5,0]]`). **No es un costo observado**; debe contrastarse con el instructor o el banco.
- **Éxito:** ROC AUC (ordena por riesgo sin fijar umbral y sirve con clases desbalanceadas; 22,1 % incumple, así que «todos pagan» acierta 77,9 % y no sirve).

## 2. Datos y limpieza

> Fuente: `reports/01_eda.md`, `src/proyecto/data/`.

- **Unidad de análisis y partición:** cada fila es un cliente distinto (30.000 `ID` únicos), por lo que la partición por fila equivale a partir por cliente. 80 % entrenamiento (24.000) y 20 % prueba (6.000), estratificada, semilla 42.
- **Hallazgos y decisiones:** sin nulos; `EDUCATION` tiene códigos 0, 5 y 6 y `MARRIAGE` el código 0 que no están en la documentación (se agrupan con una regla fija y documentada); `PAY_*` mezcla estados de pago y se trata como estado categórico.
- **Datos personales o sensibles:** no hay nombres ni documentos. `ID` se excluye del modelo. `SEX`, `AGE` y `MARRIAGE` son sensibles: se revisó su efecto por segmento y no se usan para decidir a quién negar crédito.
- **Licencia:** CC BY 4.0 (Yeh, 2009, UCI).

## 3. Variables

> Fuente: `reports/04_features.md`.

Se probaron ideas de utilización del cupo, saldo pendiente, rachas de atraso y pagos en cero. Se conserva un paquete de cinco variables (`utilization_mean`,
`pending_balance_last`, `delay_streak`, `n_zero_payments`, `log_limit`). Mejora el AUC de la logística de 0,7697 a 0,7785 y el de LightGBM de 0,7853 a 0,7870:
clara en la logística, pequeña en LightGBM. **Es optimista**: se eligió y midió con los mismos 24.000 clientes. Se **descartó** quitar `SEX` (cuesta ~0,001 de AUC) y
quitar `PAY_0` (cuesta ~0,03: dependencia fuerte del último estado de pago). La regla de conservación se cambió una vez, después de ver un resultado; el reporte muestra ambas cifras.

## 4. Torneo de modelos

> Fuente: `reports/05_torneo.md`.

Mismos 24.000 clientes y mismos 15 pliegues para todos (5 estratificados × 3 repeticiones). LightGBM, XGBoost y bosque aleatorio quedan prácticamente
empatados (0,787 ± 0,006); la regresión logística queda 0,008 por debajo (0,7785). Las líneas base: «siempre paga» 0,50 y la regla `PAY_0 ≥ 2` 0,64.
El torneo descarta: se siguió con LightGBM como principal y la logística como referencia interpretable.

## 5. Optimización y seguimiento

> Fuente: `reports/06_optimizacion.md`, `reports/06_adopcion_estefania.md`.

Estefanía ejecutó 12 ensayos de Optuna (rama `origin/Estefania`). Su propio resultado: la mejora del modelo optimizado **cruzó el cero**
(IC 95 % de la penalización: −0,0185 a +0,0072), así que la regla del curso indica quedarse con los parámetros de partida, que son más simples. El grupo adoptó esa
configuración y **no repitió la búsqueda** (`own_real_optimization_executed = False`). Seguimiento: MLflow; esta máquina no tiene conexión con Databricks, así que el
registro se hizo en MLflow local (ver sección 7).

## 6. Validación e interpretación

> Fuente: `reports/07_validacion.md` (protocolo) y `reports/07_umbral_y_resultados.md` (resultados).

Decisiones congeladas **antes** de abrir la prueba: LightGBM con paquete E, sin calibrar (no bajó Brier ni log loss), umbral de validación 1/6.

| ROC AUC | Validación cruzada (24.000) | Reserva interna (6.000) | Prueba (6.000, abierta una vez) |
|---|---|---|---|
| LightGBM | 0,787 | 0,787 | **0,782** |
| Regresión logística | 0,778 | 0,774 | 0,769 |
| Siempre «paga» | 0,500 | 0,500 | 0,500 |

En la prueba, LightGBM supera a la logística por 0,013 de AUC (intervalo 0,006 a 0,018, que excluye el cero). Con el umbral 1/6 la prueba da recall 74 %, precisión 38 %
y 43 % de clientes marcados. La brecha validación-prueba es pequeña (0,005): no hay señal de sobreajuste. **La prueba ya estaba expuesta (EDA y torneo): es una evaluación complementaria, no independiente.**

**Umbral de trabajo 0,20.** Con validación cruzada, la penalización 5:1 es casi plana cerca del óptimo, pero la carga de revisión cae con el umbral: 0,20 marca 34 % de
los clientes en vez de 43 % (−20 % de carga), con recall 68 % en vez de 75 % y +0,7 % de penalización. Elegido con la regla «el umbral más alto cuya penalización quede a menos de 2 % del mínimo».
**Transparencia:** se eligió después de ver la prueba con 1/6; se calculó solo con datos de entrenamiento y la prueba no se reabrió con 0,20.

**Interpretación:** `PAY_0` domina (baja 0,080 de AUC al permutarla), luego `LIMIT_BAL` (0,021). Asociación, no causa. **Segmentos:** el AUC se mantiene entre 0,75 y 0,80 por sexo y edad;
el tramo 45–60 años es el más débil (0,75); el de 60 o más tiene solo 51 clientes en la reserva y no permite conclusiones.

## 7. Registro del modelo

> Fuente: `reports/08_registro.md`, `reports/registro_08.json`, `models/ficha_modelo.md`.

Paquete verificable (pipeline completo, esquema, metadatos y huellas SHA-256), recuperado e idéntico al guardado (diferencia máxima de probabilidad 0,0). Registrado como
`riskvision_lightgbm_e` **versión 1, alias `champion`**, estado aprobado por el grupo (ver nota), en **MLflow local** porque no hay conexión con Databricks (`.env` sin perfil).
Para repetirlo en Databricks: completar `.env` y ejecutar `scripts/run_etapas_07_09.py --approve-definitive --register`. Exportado a `models/model.joblib` y `models/input_schema.json`.
_Nota:_ la aprobación definitiva la registró el usuario que ejecutó el flujo en nombre del grupo; conviene que el grupo la confirme por escrito.

## 8. Despliegue

> Fuente: `reports/09_despliegue.md`.

La app (`app/streamlit_app.py`) carga `models/model.joblib` desde el repositorio y **no consulta Databricks**; valida entradas (rechaza, por ejemplo, edades fuera de 18–100) y muestra aviso de uso responsable
y cómo se decide. Se probó con `streamlit.testing` (carga, evaluación y rechazo de un dato inválido sin errores). **Pendiente:** publicar en Streamlit Community Cloud, probar el enlace en ventana privada y el ensayo de 10 minutos.

## 9. Riesgos, límites y siguientes pasos

- **Qué no hace:** no estima causas, no calcula ahorro y no debe decidir solo; marcar a 1 de cada 3 clientes es mucha carga de revisión y es una decisión de negocio.
- **Límites de los datos:** un banco de Taiwán en 2005; puede no servir en otro país o época. La prueba ya estaba expuesta.
- **Uso responsable:** `SEX`, `AGE` y `MARRIAGE` son sensibles; no usarlos para negar crédito. El instructor/banco deben confirmar el costo real y la capacidad de revisión.
- **Siguiente paso concreto:** conseguir la capacidad de revisión y el costo reales, repetir el análisis de umbral con ellos y, si hay tiempo, una optimización propia y variables de tendencia de pagos. En producción vigilar la deriva de `PAY_0` y de la tasa de incumplimiento.

## Anexo: cómo reproducir el proyecto

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe -r requirements.txt
# datos originales en data/raw/; notebooks 01 a 05 generan data/processed/
python -m pytest -q                                   # pruebas
python scripts/run_etapas_07_09.py --approved-by "Nombre" --out resultados.json   # 07->09 (no abre la prueba)
streamlit run app/streamlit_app.py                    # app local
```

Notebook `notebooks/10_ejecucion_real_07_09.ipynb` resume los resultados reales de 07–09.
