# Especificación del proyecto

> Plantilla. Complétenla en el grupo y manténganla actualizada. Claude la lee antes de generar
> código: lo que escriban aquí manda sobre cualquier otra suposición.

## 1. Grupo

| Integrante | Rol (datos / modelado / app / negocio) |
|---|---|
| Juan David Ardila Arboleda | Analista de Datos |
| Lina Isabel Vasquez Villegas | Analista de Datos |
| Andres Felipe Orozco Roa | Analista de Datos |
| Cristina Garces | Analista de Datos |
| Estefania Cervera | Analista de Datos |

Nombre de la "startup": RiskVision   Repositorio (público): https://github.com/aforozco1808-dotcom/Laboratorio-de-an-lisis-avanzado-Grupo-6---RiskVision

## 2. Caso de negocio

- **Pregunta de negocio** (una frase, que alguien sin formación técnica entienda): ¿Qué clientes de tarjeta de crédito tienen alta probabilidad de no pagar el próximo mes?
- **¿Quién usará el resultado y qué decisión tomará con él?** El área de riesgo de crédito del banco. Con la probabilidad de impago de cada cliente decide si ajusta el límite de crédito, hace seguimiento preventivo o ofrece reestructurar la deuda.
- **¿Qué cuesta equivocarse?** Para priorizar revisiones preventivas, el grupo plantea la hipótesis económica de que no detectar a un cliente que incumple (falso negativo) recibe cinco veces la penalización de una falsa alarma. La relación **5 a 1 no es un costo observado del banco**. Se representa con `[[0, 1], [5, 0]]` (filas: realidad; columnas: decisión; orden: paga, incumple). Los aciertos reciben 0 como simplificación: detectar no demuestra evitar toda la pérdida ni intervenir es necesariamente gratis. No se han medido pérdidas, recuperaciones ni efectos de seguimiento, reducción de límites o reestructuración. Debe contrastarse la hipótesis con el instructor o con información económica del banco; los costos relativos no son ahorro financiero demostrado.

## 3. Datos

- **Fuente** (enlace de Kaggle, empresa, etc.): UCI Machine Learning Repository, "Default of Credit Card Clients" (clientes de un banco de Taiwán, abril a septiembre de 2005). Archivo: `data/raw/default of credit card clients.xls`.
- **Filas / columnas:** 30.000 filas y 25 columnas (24 variables más `ID`). Sin valores nulos.
- **Variable objetivo** (columna que queremos predecir o describir): `default payment next month` (1 = incumple, 0 = paga). Clase desbalanceada: 22,1 % incumple (6.636 de 30.000).
- **Tipo de problema:** ☒ clasificación  ☐ regresión  ☐ clusterización
- **¿Una misma entidad (cliente, paciente, empresa) aparece en varias filas?** ☒ no  ☐ sí → la partición
  será por: fila (cada fila es un cliente distinto; `ID` no se repite), estratificada por la variable objetivo: 80 % entrenamiento (24.000 filas) y 20 % prueba (6.000 filas), con semilla 42 y validación cruzada de 5 folds estratificados.
- **Datos personales o confidenciales presentes:** ☐ no  ☒ sí → se eliminarán/anonimizarán así: no hay nombres ni documentos, solo un `ID` anónimo. `SEX`, `AGE` y `MARRIAGE` son atributos sensibles: se revisará su efecto y no se usarán para decidir a quién se le niega crédito. `ID` se excluye del modelo.
- **Licencia o permiso de uso de los datos:** CC BY 4.0 (verificada el 2026-10-30 en la página oficial de UCI). Permite usar y redistribuir citando la fuente: Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

## 4. Métrica

- **Métrica principal:** ROC AUC (definida en `model_selection_classification_REVISION_FINAL.ipynb`)   **Por qué esa y no otra:** las clases están desbalanceadas, así que la exactitud engaña: un modelo que dice "todos pagan" acierta 77,9 % y no sirve. El ROC AUC mide qué tan bien el modelo ordena a los clientes de menor a mayor riesgo, sin fijar aún un umbral, porque el dataset no trae los costos de negocio. 0,5 equivale al azar y 1,0 a una separación perfecta.
- **Métricas secundarias:** recall de la clase "incumple" (por la penalización del falso negativo), precision, F1, AUC-PR, penalización relativa media y porcentaje de clientes marcados. Comparar modelos y líneas base sobre las mismas observaciones de validación e informar variabilidad; un ROC AUC mayor no garantiza menor costo ni impacto económico.
- **Línea base y referencias exploratorias:** modelo que siempre predice la clase más frecuente ("paga"): ROC AUC 0,5000, exactitud 77,9 % y recall 0 %. Resultados existentes del notebook de selección: Gradient Boosting inicial 0,7815 ± 0,0058 en validación cruzada; optimizado 0,7850; **evaluación previa sobre la prueba original 0,7784**. En la comparación previa todos los modelos probados superaron la línea base (KNN 0,7589, árbol de decisión 0,7589 y regresión logística 0,7272, con `PAY_*` tratado como número; como estados sube a 0,77–0,78, ver `reports/05_torneo.md`). Se conservan como referencias exploratorias, sin recalcularlas ni asumir que demuestran utilidad económica. Con la matriz 5 a 1, las referencias de costo relativo del notebook 01 son: siempre «paga» 1,11; azar 1,03; marcar a todos 0,78.
- **Umbral teórico y selección posterior:** si `p` es una probabilidad calibrada de incumplimiento, decidir «paga» cuesta `5p` y decidir «incumple» cuesta `1 - p`. El punto de indiferencia es **`1/6 ≈ 0,167`**. Es una referencia provisional que depende de calibración en la población de uso, de la matriz asumida y de no incluir otros costos o restricciones; en la igualdad hay empate. El umbral posterior se seleccionará exclusivamente mediante validación dentro del entrenamiento original, comprobando calibración y carga de revisión. La tabla de escenarios del notebook 01 conserva cifras exploratorias cuyo procedimiento de cálculo y selección de umbrales está pendiente de trazabilidad; no acredita una selección por validación cruzada.
- **Estado y protocolo de evaluación:** el EDA ya exploró el dataset completo y la prueba original ya fue evaluada en el notebook de selección. Sus métricas **no constituyen una evaluación final completamente independiente**. Las nuevas decisiones de variables, modelo, hiperparámetros, calibración y umbral se tomarán exclusivamente mediante validación dentro del 80 % original de entrenamiento. Se propone validación cruzada anidada: la selección ocurre en los pliegues internos y el procedimiento se evalúa en los externos, sin usar estos últimos para reajustar decisiones. Esto reduce el optimismo de la selección futura, pero no elimina la exposición previa a los datos. Una vez congeladas las decisiones, la prueba original podrá reportarse como evaluación complementaria con esa limitación. Una evaluación final independiente requiere datos nuevos no utilizados para decidir; cambiar la semilla no los proporciona.

## 5. Riesgos

- Fuga de información y exposición previa: escalar o imputar antes de partir y usar la prueba para elegir el umbral o el modelo introducen riesgo de optimismo. El EDA global y la evaluación previa ya expusieron la prueba original. En adelante, ajustar y seleccionar exclusivamente dentro del entrenamiento; documentar esa exposición y no presentar la prueba original como independiente.
- Sesgos o equidad: `SEX`, `AGE` y `MARRIAGE` pueden producir decisiones discriminatorias; comparar errores por grupo en la etapa de validación.
- Datos faltantes o de mala calidad: no hay nulos, pero `EDUCATION` tiene códigos 0, 5 y 6 y `MARRIAGE` el código 0, que no están en la documentación; `PAY_*` mezcla estados de pago con códigos como -2 y 0. Documentar cómo se tratan.
- Límites de lo que el modelo puede decir: los datos son de un solo banco de Taiwán en 2005; predice probabilidad de impago, no causas, y puede no servir en otro país o época.

## 6. Entregables y fechas

| Hito | Contenido | Fecha |
|---|---|---|
| H1 | Ficha aprobada, EDA y preprocesamiento reproducible | Cierre de la clase 2 (vie 30 oct) |
| H2 | Torneo, optimización con tracking, modelo registrado, repo modular, app local | Cierre de la clase 4 (jue 5 nov) |
| H3 | App desplegada y presentación (≤ 5 diapositivas) | Clase 5 (vie 6 nov) |
| H4 | Informe técnico final (hasta 8 días tras la última clase) | Domingo 15 nov |

## 7. Decisiones tomadas (agréguenlas a medida que avancen)

| Fecha | Decisión | Motivo |
|---|---|---|
| 2026-10-07 | Métrica ROC AUC para torneo, optimización y prueba final | Clases desbalanceadas y sin costos de negocio para fijar un umbral |
| 2026-10-14 | Gradient Boosting pasa a optimización (30 trials con Optuna) | Mayor ROC AUC promedio en validación cruzada; `max_depth=5` quedó en el límite del rango |
| 2026-10-14 | Montos en escala original, sin `log`; imputación por mediana y estandarización dentro del pipeline | Hay ceros y negativos en `BILL_AMT`, así que `log(x)` no sirve; evita fuga de información |
| 2026-11-05 | Subir al repositorio `data/processed/` (`train.joblib` y `test.joblib`) | Todos trabajan con la misma partición 80/20 (semilla 42) sin depender de ejecutar el notebook 03. Se sabe que expone la partición reservada de prueba en el historial público (los datos de origen ya son públicos) y que esos archivos binarios pueden cambiar si se vuelven a generar; la prueba sigue sin usarse para decidir |
| 2026-10-30 | Copiar el dataset a `data/raw/` y dejar el original en la raíz | Los compañeros lo reciben por GitHub, porque `data/raw/` no se sube |
| 2026-10-31 | Etapa 4: paquete de 5 variables (`util_media`, `saldo_pend_1`, `racha_atraso`, `n_pago_cero`, `log_limite`) y regla de decisión por mejora pareada, en lugar de mejora / ruido | La regla original no confirmó el paquete en una de dos repeticiones con otros pliegues, aunque la mejora fue casi idéntica y ganó en los 45 pliegues; el cambio se decidió después de ver el resultado y está documentado en `reports/04_features.md` |
| 2026-10-31 | Precisar la hipótesis 5 a 1, distinguir umbral teórico de selección por validación y reconocer la exposición de la prueba original | Las penalizaciones no son costos observados ni ahorro demostrado; las métricas previas se conservan como exploratorias y la prueba consultada no es una evaluación final independiente |
| 2026-11-06 | Etapa 06: se adopta el LightGBM de Estefanía (paquete E) sin repetir Optuna | Su mejora optimizada cruzó el cero; los parámetros de partida son más simples (`reports/06_adopcion_estefania.md`) |
| 2026-11-06 | Etapas 07–09: sin calibración; decisión congelada con umbral 1/6 antes de abrir la prueba (abierta una sola vez; AUC 0,782) | La calibración no bajó Brier ni log loss; la prueba ya estaba expuesta, por lo que es una evaluación complementaria (`reports/07_umbral_y_resultados.md`) |
| 2026-11-06 | Umbral de trabajo de la app = 0,20 | La penalización 5:1 es casi plana cerca del óptimo y 0,20 baja 20 % la carga de revisión. Se calculó con validación cruzada, pero **después** de ver la prueba con 1/6: queda declarado |
| 2026-11-06 | Registro del modelo (v1, alias champion) y del torneo en MLflow local | No hay conexión con Databricks en esta máquina; el curso permite el respaldo local. Pendiente repetirlo en Databricks (`.env`) |
