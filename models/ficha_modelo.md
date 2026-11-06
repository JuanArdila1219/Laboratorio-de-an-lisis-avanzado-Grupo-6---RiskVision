# Ficha del modelo

> Estado: modelo **aprobado como definitivo por el usuario que ejecutó el flujo, en nombre del grupo (pendiente de confirmación escrita del grupo)** y registrado como `riskvision_lightgbm_e` v1, alias `champion`, en **MLflow local** (`reports/registro_08.json`). Falta el registro en Databricks (no hay conexión en esta máquina).
> Cifras y su motivo: `reports/07_umbral_y_resultados.md`.

## Qué es

Estima la probabilidad de que un cliente de tarjeta de crédito **no pague el próximo mes**, para que el área de riesgo
de crédito priorice revisiones preventivas. Apoya una decisión; la decisión final es de una persona.

- **Modelo:** LightGBM con el paquete de variables E (uso del cupo, saldo pendiente, racha de atrasos, pagos en cero, log del límite). Parámetros adoptados del experimento de Estefanía; no hubo optimización propia.
- **Registro:** `riskvision_lightgbm_e` versión 1, alias `champion`, en MLflow local (Databricks pendiente). Copia para la app: `models/model.joblib`.
- **Regla de decisión:** se marca «riesgo» si la probabilidad de incumplir es **≥ 0,20** (punto de trabajo elegido con validación cruzada; el teórico 5 a 1 sería 0,167).

## Datos de entrenamiento

- **Fuente y licencia:** UCI «Default of Credit Card Clients» (banco de Taiwán, 2005), CC BY 4.0. Yeh, I. (2009). https://doi.org/10.24432/C55S3H
- **Tamaño:** 24.000 clientes de 30.000 (cada fila es un cliente distinto).
- **Partición:** por cliente (fila), estratificada, 80/20, semilla 42. La prueba ya había sido expuesta en el EDA y el torneo; se abrió una vez para la evaluación complementaria.
- **Datos personales:** no hay nombres ni documentos, solo un `ID` anónimo que se excluye del modelo.

## Rendimiento

| Métrica | Validación cruzada (5 pliegues, 24.000) | Prueba, umbral 1/6 (6.000) |
|---|---|---|
| ROC AUC | 0,787 | 0,782 |
| ROC AUC, siempre «paga» | 0,500 | 0,500 |
| ROC AUC, regresión logística | 0,778 | 0,769 |

Con el umbral 0,20 de la app, en validación cruzada: marca 34 % de los clientes, detecta 68 % de quienes incumplen y
acierta 44 % de lo que marca. **La prueba no es independiente** (ya había sido expuesta) y no se calcularon intervalos
del 95 % por métrica; solo para la diferencia con la logística.

## Variables que más pesan

`PAY_0` (estado del último pago) domina; después `LIMIT_BAL`, `BILL_AMT1` y `BILL_AMT2`. Es una asociación del modelo, no una causa.

## Límites y usos no recomendados

- Datos de un solo banco de Taiwán en 2005: puede no servir en otro país o época.
- La relación de costo 5 a 1 y el umbral son **hipótesis del grupo**, sin costos medidos ni ahorro demostrado.
- `SEX`, `AGE` y `MARRIAGE` son atributos sensibles. El segmento de 45 a 60 años tiene el peor AUC (0,75) y el de 60 o más tiene solo 51 clientes en la reserva.
- **No debe usarse** para negar crédito de forma automática ni para decidir sobre personas sin revisión humana.
- **No se evaluó:** estabilidad en el tiempo, otras poblaciones ni deriva.

## Reproducibilidad

Código en `src/`, pruebas en `tests/`, notebooks en `notebooks/`, recorrido completo en `scripts/run_etapas_07_09.py`.
