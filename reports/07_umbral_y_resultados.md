# 07 · Resultados reales y decisión del umbral — RiskVision, Grupo 6

> Complementa `reports/07_validacion.md` (protocolo). Cifras reproducibles con `scripts/run_etapas_07_09.py`.
> Estado: modelo **aprobado como definitivo por el usuario que ejecutó el flujo, en nombre del grupo (pendiente de confirmación escrita del grupo)** y registrado como `riskvision_lightgbm_e` v1, alias `champion`, en **MLflow local** (`reports/registro_08.json`). Falta el registro en Databricks (no hay conexión en esta máquina).

## 1. Qué se validó (solo con el entrenamiento, 24.000 clientes)

LightGBM con las variables E y los parámetros adoptados de Estefanía (`reports/06_adopcion_estefania.md`).
Validación cruzada de 5 pliegues (OOF) sobre los 24.000 y, además, una reserva interna de 6.000 clientes (el modelo se ajusta con los otros 18.000). Calibración: no se aplica porque
no bajó Brier ni log loss (regla escrita antes de ver los números).

| ROC AUC | Validación cruzada, 24.000 | Reserva interna, 6.000 (modelo ajustado con los otros 18.000) | Prueba, 6.000 (abierta una sola vez, umbral 1/6) |
|---|---|---|---|
| LightGBM | 0,787 | 0,787 | 0,782 |
| Regresión logística (referencia) | 0,778 | 0,774 | 0,769 |
| Siempre «paga» (línea base) | 0,500 | 0,500 | 0,500 |

En la prueba, LightGBM supera a la logística por 0,013 de AUC (intervalo 0,006 a 0,018, que excluye el cero).
**Limitación:** la prueba ya estaba expuesta (EDA y torneo); no es una evaluación independiente.
Variable dominante: `PAY_0` (baja 0,080 de AUC al permutarla), luego `LIMIT_BAL` (0,021).

## 2. Decisión del umbral (y por qué cambió de 1/6 a 0,20)

El punto teórico de la hipótesis 5 a 1 es 1/6 ≈ 0,167. Con validación cruzada (sin la prueba), la penalización
5:1 es casi plana cerca del óptimo, pero la carga de revisión cae rápido al subir el umbral:

| Umbral | % de clientes marcados | Recall | Precisión | Penalización 5:1 | Variación vs. 1/6 |
|---|---|---|---|---|---|
| 1/6 = 0,167 | 43,2 % | 74,8 % | 38,3 % | 0,546 | — |
| **0,20** | **34,4 %** | **67,8 %** | **43,6 %** | **0,550** | **+0,7 %** |
| 0,25 | 26,8 % | 60,4 % | 49,8 % | 0,573 | +5,0 % |
| 0,50 (clase más probable) | 12,3 % | 37,4 % | 67,5 % | 0,732 | +34 % |

**Regla elegida:** el umbral más alto de la grilla (pasos de 0,05 más 1/6) cuya penalización 5:1 quede a menos del 2 %
del mínimo observado (0,540, en 0,15). Resulta **0,20**: revisa 9 puntos porcentuales menos de clientes que 1/6
(−20 % de carga) a cambio de perder 7 puntos de recall.

**Es un punto de trabajo, no una verdad económica.**
- La relación 5 a 1 es una hipótesis del grupo; no hay costos medidos. Con otra relación el umbral cambia
  (óptimo en OOF: 0,325 para 2:1; 0,245 para 3:1; 0,09 para 10:1).
- Si el área de riesgo tiene una capacidad fija de revisión, conviene elegir el umbral por capacidad: revisar el
  10 % da recall 32 %, el 20 % da 52 % y el 30 % da 64 %.
- **Transparencia:** esta elección se hizo *después* de conocer las métricas de la prueba con 1/6. Se calculó solo con
  OOF de entrenamiento y la prueba no se volvió a abrir, pero no es una decisión congelada antes de verla. Las
  métricas de prueba del apartado 1 corresponden a 1/6; con 0,20 no se calcularon.

## 3. Coherencia con la app

El esquema `models/input_schema.json` declara `decision_threshold = 0,20` y `positive_class = "1"`. La app marca
«riesgo» cuando la probabilidad llega a ese valor (antes decidía por la clase más probable, o sea 0,5, que habría
dado un recall de ~37 %). Prueba automática: `tests/test_modelo_exportado.py`.
