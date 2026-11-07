---
marp: true
theme: default
paginate: true
header: "RiskVision · Grupo 6"
footer: "Casos Avanzados · Maestría en Ingeniería Analítica"
---

# 1 · De cada 100 clientes, 22 dejarán de pagar

**Y hoy el banco no sabe cuáles.**

- Un analista de riesgo (caso ilustrativo) tiene tiempo para **seguir de cerca a unos cuantos clientes**, no a todos.
- Si elige mal, pierde en dos sentidos: **no ve** al cliente que dejará de pagar, o **molesta** a uno que sí pagaba.
- Equivocarse en el primer sentido nos parece **5 veces más grave** que en el segundo. _(Hipótesis del grupo; no es un costo medido del banco.)_

> **La pregunta:** ¿a qué clientes miramos primero?

<!-- Notas (2 min): presentar al analista, el dilema de "a quién reviso" y la hipótesis 5 a 1 como supuesto, no como dato. -->

---

# 2 · Dejamos que los datos ordenen a los clientes

- **30.000 clientes** de un banco de Taiwán (2005). Guardamos **6.000 aparte** para confirmar al final (ya se habían explorado antes y lo declaramos).
- **Pistas que descubrimos:** el último estado de pago pesa más que cualquier otra variable; sumamos 5 señales de comportamiento (uso del cupo, saldo pendiente, rachas de atraso).
- **Competencia entre modelos:** 4 modelos y 3 reglas simples, con las mismas reglas del juego. Ganó **LightGBM**; la regresión logística quedó cerca, pero por debajo.
- **Lo que descartamos:** repetir la búsqueda de parámetros, porque su mejora no se distinguía del ruido.

> **Moraleja:** preferimos el modelo sencillo que se puede defender al que "parece" mejor.

<!-- Notas (parte del minuto 2 y del bloque de resultados): sin nombres de algoritmos en el titular. -->

---

# 3 · La historia en números

De cada 100 clientes: **22 dejarán de pagar**.

| | Sin el modelo | Con el modelo |
|---|---|---|
| Clientes que el analista marca | 0 | **34** |
| De los 22 que fallarán, ve a… | 0 | **15 (68 %)** |
| De los 34 marcados, aciertan… | — | **15 (44 %)** |

- El modelo ordena bien: **ROC AUC 0,79** (0,50 sería tirar una moneda).
- Fijamos el corte en **20 % de probabilidad**: revisar 9 clientes menos por cada 100 que con el corte "teórico" (43 → 34), a cambio de perder de vista a 1 o 2 de los 22 que fallarán.

_Nota técnica:_ AUC 0,787 en validación cruzada y 0,782 en la prueba. La prueba ya se había expuesto antes, así que no es una evaluación independiente. Los números por cada 100 son los de validación cruzada.

<!-- Notas (3 min): contar la cifra, luego el motivo del corte; el costo 5 a 1 no es ahorro en dinero. -->

---

# 4 · Veámoslo funcionar

**Enlace de la app:** _pendiente de publicar en Streamlit Community Cloud_ (hoy corre con `streamlit run app/streamlit_app.py`).

| Caso | Qué pasa |
|---|---|
| **Normal:** cliente de bajo riesgo | Probabilidad 2 % → **no marcar** |
| **Límite:** justo en el corte | Probabilidad 20 % → **marcar** (cambia la decisión) |
| **Alto riesgo** | Probabilidad 89 % → **marcar** |
| **Inválido:** edad 5 años | La app **lo rechaza**: «AGE: debe estar entre 18 y 100» |

La app explica en pantalla cómo decide y recuerda que **la decisión final es de una persona**.

> Plan B: captura de pantalla con fecha y la app en local.

<!-- Notas (4 min): abrir la app, no leer la diapositiva. Probar el enlace antes en ventana privada. -->

---

# 5 · Lo que el modelo NO dice, y qué sigue

- **No explica causas ni calcula ahorro.** Y no debe negar crédito por sí solo.
- **Marca a 1 de cada 3 clientes:** es mucha carga de revisión. Es una decisión de negocio, no del modelo.
- **Datos de un solo banco y de 2005:** puede no servir en otro país o época. Vigilamos la equidad: el tramo de 45 a 60 años rinde algo menos (AUC 0,75).
- **Si tuviéramos más tiempo:** pediríamos al banco el **costo real** y su **capacidad de revisión** para recalcular el corte; después, una optimización propia y variables de tendencia de pagos.
- **Una decisión que cambiaríamos:** fijar el corte *antes* de abrir la prueba. Lo elegimos después de verla (con datos de entrenamiento) y lo declaramos.

<!-- Notas (1 min): cierre honesto; quién responde preguntas de negocio y quién las técnicas. -->
