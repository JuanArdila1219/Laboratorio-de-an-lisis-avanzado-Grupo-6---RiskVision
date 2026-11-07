# Guía para defender el código (criterio D: aporte individual)

> El evaluador elige un fragmento generado con IA y pregunta a un integrante distinto del que expuso. Esta guía explica, en palabras simples,
> las piezas de las etapas 07–09 para que cualquiera del grupo pueda responder. **No reemplaza entender el código**: ábranlo mientras la leen.

## Idea general (30 segundos)
Entrenamos un modelo que da la **probabilidad** de que un cliente deje de pagar. Primero lo validamos solo con datos de entrenamiento, **congelamos** las decisiones,
entrenamos la versión final, la guardamos con huellas de seguridad, la registramos y la app la usa con un **corte** (20 %) para decidir a quién marcar.

## Piezas y qué responder
| Archivo | Qué hace | Pregunta típica y respuesta corta |
|---|---|---|
| `src/proyecto/models/validation.py` · `crossfit_probabilities` | Para cada cliente, predice con un modelo que **no lo vio** (5 pliegues) | *¿Por qué no entrenar y evaluar con todo?* Porque daría un resultado optimista; así cada predicción es "honesta" |
| `validation.py` · `select_threshold` | Busca el corte que minimiza la penalización 5:1 usando solo esas predicciones | *¿Se usó la prueba?* No. La prueba nunca decide nada |
| `validation.py` · `freeze_for_evaluation` | Fija calibración y corte **antes** de abrir la prueba | *¿Para qué?* Para que la prueba solo confirme y no "enseñe" |
| `src/proyecto/models/decision.py` | Contrato 07→08 con responsable, fecha y huella | *¿Qué evita?* Cambiar la configuración sin dejar rastro |
| `src/proyecto/models/persistence.py` · `save_bundle`/`load_bundle` | Guarda el pipeline completo con SHA-256 y lo recupera verificando | *¿Por qué guardar el pipeline y no solo el modelo?* Porque el preprocesamiento aprendido viaja con él |
| `src/proyecto/models/final_run.py` | Ordena los pasos 07→09 | *¿Quién decide la aprobación?* Una persona; el código no la deduce de las métricas |
| `src/proyecto/inference/predict.py` · `decide_by_threshold` | Marca "1" si la probabilidad llega al corte | *¿Por qué 20 % y no 50 %?* El error de no detectar pesa más (hipótesis 5:1); 50 % dejaría pasar a muchos |
| `predict.py` · `validate_inputs` | Rechaza datos imposibles (p. ej. edad 5) con un mensaje claro | *¿Qué pasa si falta un dato?* El pipeline lo imputa; si es imposible, se rechaza |
| `app/streamlit_app.py` | Formulario que carga `models/model.joblib` | *¿Consulta Databricks?* No (regla 8) |

## Cinco cosas que todo integrante debe poder decir
1. La métrica principal es **ROC AUC** (ordena por riesgo sin fijar corte) y el modelo logra 0,787 en validación cruzada y 0,782 en la prueba.
2. La prueba **ya estaba expuesta** (EDA y torneo): no es una evaluación independiente.
3. El costo **5 a 1 es una hipótesis**, no un costo medido; el corte 0,20 también.
4. El corte 0,20 se eligió **después** de ver la prueba con 1/6, aunque se calculó con datos de entrenamiento. Está declarado.
5. El modelo apoya una decisión humana; **no** debe negar crédito automáticamente.

## Cómo verificar código generado por IA (checklist rápido)
- ¿Todo el preprocesamiento está dentro del `Pipeline` y se ajusta solo con entrenamiento?
- ¿Alguna decisión (modelo, corte, parámetros) se tomó mirando la prueba?
- ¿Hay datos personales, tokens o rutas locales en lo que se sube?
- ¿Cada función de `src/` que transforma datos tiene su prueba en `tests/`? (`python -m pytest -q`)
