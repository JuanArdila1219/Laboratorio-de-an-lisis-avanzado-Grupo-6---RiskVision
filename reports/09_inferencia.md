# 09 · Inferencia de RiskVision — Grupo 6

## Estado y alcance

Implementación preparada para diagnóstico sintético y posterior integración con una interfaz. LightGBM original con paquete E sigue **PROVISIONAL**. No existe todavía un modelo definitivo entrenado, evaluado y aprobado. Esta etapa no ejecuta entrenamiento, Optuna, consultas a datasets, registros remotos ni despliegue. No se utilizan resultados de otra rama como evidencia propia.

Rama de desarrollo: revision-h2-riskvision. Cada sesión registra la revisión efectiva del código y el SHA-256 del artefacto; no se inventa un commit para cambios sin publicar. Las métricas reales permanecen pendientes: las pruebas de software no demuestran rendimiento predictivo.

## Funcionamiento y API reutilizable

El flujo es: cliente ya limpio → validación del contrato → variables E, imputación y codificación del pipeline guardado → probabilidad de incumplimiento → clasificación con umbral explícito → resultado interpretable y trazable.

open_session recibe el directorio del paquete 08, trusted=True, la huella externa manifest_sha256 y los controles. Delega la carga en load_bundle: verifica integridad antes de deserializar, confirma versiones compatibles, esquema, estado y pipeline completo ajustado. No carga un Joblib aislado, no busca champion y no cambia de modelo si falla una comprobación.

predict_clients acepta un diccionario para un cliente o un DataFrame para un lote. Conserva el orden e índice de filas, incluso índices repetidos. Devuelve default_probability, classification, result, threshold, threshold_source, model_name, model_version, model_status, artifact_sha256 y diagnostic_only. Las advertencias, alcance, identidad del artefacto y explicación de límites quedan en attrs. La clasificación es 1 cuando la probabilidad es mayor o igual al umbral.

No hay selección de umbral en 09. En diagnóstico se exige un valor sintético explícito, por ejemplo 0.5, marcado como no operativo. Para uso real se recupera exclusivamente decision.settings[decision.model].threshold, respaldado por aprobación definitiva y configuración congelada. Se rechaza un umbral sintético en modo real y cualquier cambio del umbral aprobado. Ni 0.5 ni 1/6 se declaran definitivos. La relación 5:1 permanece como hipótesis económica; no implica ahorro demostrado.

## Modelo requerido y compatibilidad con 08

Se utiliza el paquete compuesto por pipeline.joblib, input_schema.json, metadata.json y manifest.json. El pipeline conserva transformaciones, imputadores, codificadores y modelo ajustado, incluido el calibrador cuando forme parte del artefacto. Nada se vuelve a ajustar con clientes de inferencia.

Los metadatos y huellas proceden de 08; el esquema no se reconstruye con nuevos datos. El nombre y versión informados son los del paquete local, no una versión de Model Registry inventada. Para uso real hace falta estado approved_definitive, alcance real y aprobación externa documentada con responsable, fecha e informe de validación. Los controles por sí solos nunca aprueban un provisional.

## Esquema y validaciones

Exactamente 23 columnas: LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE; PAY_0, PAY_2 a PAY_6; BILL_AMT1 a BILL_AMT6; PAY_AMT1 a PAY_AMT6. Se restablece el orden del ajuste. Se rechazan columnas faltantes, adicionales o duplicadas, incluidos ID, objetivo y variables derivadas externas.

El contrato es **predictores originales ya limpios**. Se reutilizan las reglas y dominios de 08: SEX en 1–2, EDUCATION en 1–4 y MARRIAGE en 1–3, después de las agrupaciones fijas existentes; PAY en -2, -1, 0 o 1–9; edad entera de 18 a 100; límite positivo; pagos no negativos. Se conservan facturas negativas plausibles. No se introducen nuevos límites superiores monetarios basados en la muestra.

Solo se aceptan valores numéricos reales y faltantes, no texto numérico, booleanos ni infinitos. Una columna ausente es un error; un valor ausente dentro de una columna existente puede ser None, pd.NA o NaN. Tras validar tipos, esas representaciones se normalizan a NaN numérico y la imputación queda en el pipeline guardado. No se estiman medianas nuevas. Las categorías válidas ausentes del ajuste y los valores fuera del rango observado generan advertencias. Una entrada vacía se rechaza.

La interfaz futura debe obtener o preparar datos de acuerdo con la limpieza fija del proyecto; este servicio no convierte silenciosamente códigos crudos raros ni acepta entradas imposibles como si fueran faltantes. La probabilidad no equivale a certeza, causalidad o aprobación automática de crédito.

## Ejemplos sintéticos

synthetic_clients devuelve dos clientes artificiales sin leer datos ni entrenar. Para abrir una sesión diagnóstica se necesita un paquete sintético provisional previamente generado en un temporal por las pruebas de 08. No se genera ese modelo dentro de 09.

```python
controls = InferenceControls(diagnostic_mode=True)
session = open_session(directorio_temporal_08, trusted=True,
    manifest_sha256=huella_externa_08, controls=controls, synthetic_threshold=0.5)
clientes = synthetic_clients()
uno = predict_clients(clientes.iloc[0].to_dict(), session)
lote = predict_clients(clientes, session)
```

En el notebook, DIAGNOSTIC_MODE y RUN_SYNTHETIC_EXAMPLES se activan solo para este escenario; deben configurarse explícitamente directorio, origen confiable, huella y umbral sintético. Los ejemplos cubren fila individual, lote, faltante y edad imposible. La configuración original conserva todos los indicadores en False y las rutas en None.

## Controles de seguridad

DIAGNOSTIC_MODE=False, ALLOW_REAL_INFERENCE=False y ALLOW_REAL_PREDICTION_PUBLICATION=False por defecto; RUN_SYNTHETIC_EXAMPLES también está desactivado. ALLOW_TRAINING=False y ALLOW_TEST_ACCESS=False son prohibiciones permanentes de esta etapa: intentar activarlas provoca un error antes de acceder a archivos. No habilitan rutas de entrenamiento ni evaluación. Los controles se aplican en el módulo además del notebook. No se mezclan permisos reales con diagnóstico. Un artefacto sintético solo se utiliza en diagnóstico y nunca se publica como definitivo.

El notebook no ofrece rutas de entrenamiento ni lectura de conjuntos de datos. Las pruebas prohíben ajustes y escrituras durante su ejecución y solo permiten deserializar los bytes verificados del modelo sintético. El notebook tampoco imprime ni exporta automáticamente resultados reales. publish_predictions requiere autorización separada, sesión real aprobada y procedencia del resultado compatible; escribe un archivo nuevo en un destino explícito, sin sobrescribir ni crear carpetas. No publica en servicios externos. En esta entrega se comprueba su bloqueo; no se publican predicciones reales.

SHA-256 detecta alteraciones, no autentica al remitente. La procedencia confiable y la aprobación son declaraciones respaldadas por el proceso de revisión, no firmas criptográficas. Pickle/Joblib solo se debe cargar de una fuente confiable. El llamador es responsable de no introducir clientes reales en modo sintético: su origen no se puede deducir matemáticamente de sus valores.

## Evidencia de pruebas

Pruebas exclusivamente sintéticas: carga del paquete 08; inferencia individual y lote; probabilidades válidas y equivalentes al pipeline; frontera de clasificación; orden de filas y columnas; faltantes; tipos y dominios inválidos; columnas erróneas; integridad; permisos; recuperación en un proceso Python nuevo; notebook completo con controles desactivados y con diagnóstico explícito. La aprobación real se simula únicamente en memoria para comprobar el contrato de umbral y bloqueo de exportación; no constituye una aprobación del modelo.

Comprobaciones locales del 6 de noviembre de 2026: regresión de 385 pruebas aprobada, excluyendo test_tuning.py y test_reporting.py para no ejecutar Optuna. Después de completar los controles permanentes, la verificación final de la etapa 09 aprobó sus 43 pruebas específicas, incluida la ejecución completa del notebook tanto bloqueado como en diagnóstico. Los recuentos se solapan y no deben sumarse. La regresión mostró una advertencia de deprecación de SQLAlchemy en Tracking local, sin fallos. Notebook JSON válido, 15 celdas y metadatos conservados, y git diff --check sin errores. Se detectó y corrigió la conversión de pd.NA a NaN y se evitó una advertencia falsa por códigos equivalentes como 1 y 1.0. No hay métricas reales en este informe.

## Limitaciones y pendientes en Databricks

Reutilizar requirements-serverless-06.txt desde Environment, reiniciar la sesión y ejecutar el notebook desde la primera celda. No se añaden dependencias. Configurar RISKVISION_PROJECT_ROOT, revisión del código si Git no está disponible, ruta del paquete (por ejemplo un Volume autorizado), huella externa y origen confiable. Reproducir las versiones registradas por 08; el cargador exige las bibliotecas principales exactas y Python de la misma versión mayor/menor. El manifiesto de dependencias no constituye un lock completo del entorno base y dependencias transitivas.

Primero verificar el diagnóstico con un paquete sintético en un temporal del entorno. Confirmar imports, carga desde otra sesión, predicción individual/lote, errores y ausencia de publicación. Después completar la evaluación real 07, documentar aprobación y umbral, persistir el paquete real en 08 y solicitar autorización para inferencia real y, por separado, exportación. La prueba original ya fue consultada y no es una evaluación completamente independiente; no se utiliza aquí para seleccionar nada.

No se modificaron los notebooks 01–08, no se desplegó una aplicación y no se verificó una ejecución integral remota de 06–09. El contrato del servicio permite integrar una interfaz más adelante, sin duplicar persistencia o transformaciones; la aplicación existente todavía requiere adaptación para utilizar este servicio seguro.

## Continuidad con la decisión formal del equipo

La contribución de Estefanía y own_real_optimization_executed=False viajan en los metadatos del paquete 08. La inferencia usa el umbral y calibración aprobados de la configuración congelada; no el umbral 1/6 de su experimento ni sus métricas como validación propia. Para paquetes nuevos se comprueba el contrato y la concordancia de esquema/calibración al recuperar el pipeline. El informe 09 identifica el informe 07 y refleja autorización efectiva de la sesión. Los controles siguen False; no hay entrenamiento, acceso a prueba ni aprobación desde 09.
