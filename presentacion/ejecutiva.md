---
marp: true
theme: default
paginate: true
header: "Presentación ejecutiva · RiskVision"
footer: "Casos Avanzados · Maestría en Ingeniería Analítica"
---

# 1 · Problema

- **¿Quién decide?** El área de riesgo de crédito de un banco.
- **¿Qué decide con el resultado?** A qué clientes de tarjeta hacerles seguimiento preventivo, ajustar el límite u ofrecer reestructurar la deuda.
- **¿Cuánto cuesta equivocarse?** Dejar pasar a un cliente que no pagará cuesta más que una falsa alarma. Suponemos **5 veces más** (hipótesis del grupo, no un costo medido).

> **Detectar a tiempo a quien dejará de pagar, sin revisar a todos.**

---

# 2 · Enfoque

- **Los datos:** 30.000 clientes de un banco de Taiwán (2005), 22 % incumple. Usamos 24.000 para construir y validar el modelo y 6.000 para una comprobación final.
- **Cómo se validó con honestidad:** cada fila es un cliente distinto, así que se parte por cliente; todo el preprocesamiento se aprende solo con los datos de entrenamiento.
- **Qué se comparó y qué se eligió:** 5 modelos y 3 reglas simples. Elegimos LightGBM; la regresión logística quedó 0,008 por debajo.
- **Qué se descartó y por qué:** quitar el sexo del modelo (casi no cambia, pero se vigila la equidad) y repetir la búsqueda de parámetros (la mejora de Estefanía no se distinguía del ruido).

---

# 3 · Resultado y costo de negocio

| | Sin el modelo | Con el modelo |
|---|---|---|
| Capacidad de ordenar clientes por riesgo (ROC AUC) | 0,50 (azar) | **0,78** |
| Clientes que incumplen y detectamos | 0 % ("todos pagan") | **68 %** |
| Clientes que se marcan para revisar | 0 % | **34 %** |

- De cada 100 clientes marcados, **44** realmente incumplen (la tasa general es 22).
- Subir el umbral de 1/6 a 0,20 baja la revisión **20 %** a cambio de perder 7 puntos de detección.
- _Nota técnica:_ AUC 0,787 en validación cruzada y 0,782 en la prueba. La prueba ya había sido expuesta, así que no es una evaluación independiente. El costo 5 a 1 no es dinero ni ahorro demostrado.

---

# 4 · Demo de la app

**Enlace:** _pendiente de publicar en Streamlit Community Cloud_ (en local: `streamlit run app/streamlit_app.py`).

1. **Un caso normal:** cliente al día en todos los meses → resultado «0» (no marcar).
2. **Un caso límite:** cliente con retrasos recientes y pagos bajos → cambia a «1» (marcar) cuando su probabilidad llega a 20 %.
3. **Un dato inválido:** edad 5 → la app lo rechaza: «AGE: debe estar entre 18 y 100».

> Plan B: captura de pantalla con fecha y la app en local.

---

# 5 · Riesgos y siguientes pasos

- **Qué NO hace el modelo:** no explica causas, no calcula ahorro y no debe decidir solo ni negar crédito automáticamente.
- **Límites de los datos:** un solo banco de Taiwán en 2005; puede no servir en otro país o época. Marca a 1 de cada 3 clientes: es mucha carga de revisión.
- **Uso responsable:** sexo, edad y estado civil son variables sensibles; no se usan para decidir a quién negar crédito. El tramo de 45 a 60 años rinde algo menos (AUC 0,75).
- **Siguientes pasos:** confirmar con el banco el costo real y la capacidad de revisión, y recalcular el umbral; vigilar en producción el retraso de pago (`PAY_0`) y la tasa de incumplimiento.
