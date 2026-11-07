# Preparación de Streamlit — RiskVision

## Base y alcance

Rama revision-streamlit-riskvision desde origin/main 08abd1d. main y revision-h2-riskvision no se modifican. Se reutiliza el modelo de Andrés; no hay entrenamiento, Optuna, lectura de datasets ni nuevas métricas de evaluación. La confirmación de Andrés de 0,20 autoriza configurar el punto de trabajo, no concede aprobación definitiva al modelo.

## Única configuración activa de inferencia de la app

models/input_schema.json contiene decision_threshold=0.20 y positive_class=1, más la procedencia de Andrés y el commit de publicación. app_runtime valida esa regla y la app la usa para clasificar y mostrar el corte. final_run.OPERATING_THRESHOLD ahora se deriva del mismo JSON, sin otro literal configurable. No se usa como corte el máximo de probabilidades (0,5), review_threshold (auxiliar) ni el teórico 1/6.

Se conserva el Joblib de 540.463 bytes, SHA-256 96061e2f63c6e61032843a30c5ad8640eeb25aab763dae2baa3021e84f332c5f. La carga verifica huella, Python 3.12, seis versiones científicas y concordancia de esquema/clases antes de puntuar. SHA no autentica un origen desconocido: la confianza es el repositorio revisado de Andrés; no se cargan archivos subidos por usuarios.

El formulario entrega códigos numéricos; se reutilizan los dominios y validaciones del proyecto. Se rechazan categorías inválidas, booleanos, textos numéricos fuera de categorías, infinitos, edades inválidas, límites no positivos y pagos negativos; se permiten facturas negativas plausibles y faltantes para los imputadores ya aprendidos. No se aprenden transformaciones nuevas.

## Inventario del umbral anterior (conservado)

- models/ficha_modelo.md: tabla histórica a 1/6; models/input_schema.json: historical_evaluation_threshold dentro de procedencia, sin efecto operativo.
- reports/resultados_07_09.json: settings y métricas/segmentos con threshold=0.16666666666666666. No se modifica este archivo ni sus cifras. operating_threshold es una anotación posterior, no una reevaluación de sus métricas.
- reports/07_umbral_y_resultados.md, informe_tecnico.md y guia_defensa_codigo.md: comparación histórica y declaración del cambio posterior. No se atribuyen sus métricas de prueba a 0,20.
- reports/00_caso.md, 01_eda.md, 04_features.md, 05_torneo.md y 06_adopcion_estefania.md: referencia económica o resultados exploratorios previos.
- reports/07_validacion.md y 09_inferencia.md: protocolo y contratos anteriores, no configuración activa de esta exportación Streamlit.
- src/proyecto/models/evaluate.py: THEORETICAL_THRESHOLD y valores predeterminados de métricas; validation.py los reutiliza para análisis, bootstrap y segmentos; decision.py documenta el experimento de Estefanía; tournament.py tiene una zona teórica; summary.py etiqueta la prueba a 1/6. No se cambian retrospectivamente.
- Notebooks: 01 celda 10; 02 celdas 32,38,39; 04 celdas 8,18; 05 celdas 2,7,9,19,21,23,28; 07 celdas 6,7; 09 celda 5 (numeración desde uno). No se ejecutan ni modifican.
- especificacion-proyecto.md mantiene la explicación teórica; tests/test_evaluate.py y tests/test_summary.py mantienen verificaciones históricas. La presentación y el material del curso no son entradas de configuración de la app.

El servicio de paquetes completos de notebook 09 conserva la exigencia del umbral congelado del paquete. No se sustituye silenciosamente por 0,20: falta recuperar y actualizar mediante revisión el paquete de models/_local/ si se desea unificar también esa ruta. La app usa explícitamente la exportación de Andrés y su política documentada, con aviso de demostración.

## Aprobación y evidencia pendientes

El JSON de primera corrida sigue provisional y el recibo posterior declara approved_definitive; falta confirmación escrita del grupo. El contrato original conservaba 1/6 y no se reescribe ni se modifica su huella. La ficha y la app explican que el uso es académico. No se registra, promueve ni aprueba un modelo en esta tarea.

La documentación histórica menciona OOF de 24.000 mientras que resultados_07_09.json contiene n=18.000 para esas métricas. Es una discrepancia pendiente de aclaración con Andrés; no se recalculan ni sustituyen cifras. La decisión de 0,20 fue posterior a conocer la prueba con 1/6; no se presenta como elección congelada antes de observarla ni se promete rendimiento en clientes reales.

## Dependencias y despliegue posterior

app/requirements.txt fija solamente Streamlit y las seis bibliotecas científicas del modelo. Community Cloud prioriza el archivo junto al entrypoint sobre requirements.txt de la raíz; el archivo general permanece intacto. No se requieren credenciales ni conexión a Databricks, MLflow, Optuna o los datos originales para la app.

Después de revisión y autorización de publicación: seleccionar repositorio del grupo, rama publicada de esta preparación y app/streamlit_app.py como entrypoint; Python 3.12 en Advanced settings. Verificar logs de instalación, carga y casos sintéticos en ventana privada antes de compartir el enlace. No ejecutar scripts/run_etapas_07_09.py para desplegar: ese script vuelve a entrenar. No se desplegó durante esta preparación.

Comprobación terminada: entorno nuevo creado con Python 3.12.7 en un temporal externo, instalado desde app/requirements.txt (más pytest para pruebas), sin reutilizar site-packages de .venv. pip check sin conflictos. 21 pruebas de app aprobadas en ese entorno, incluyendo carga real del artefacto, entradas artificiales individuales y por lote, probabilidades, frontera 0,20, categorías, faltantes, errores, AppTest y servidor HTTP local 127.0.0.1; servidor terminado al finalizar. Otras 8 pruebas de funciones puras de final_run aprobadas en .venv. No se ejecutó fit ni se recalcularon métricas. No hay cambios de contenido en el modelo, resultados históricos, registro o requirements.txt general. git diff --check sin errores. No demuestra compatibilidad Linux/Community Cloud hasta probar allí la instalación y el enlace público. Los rangos numéricos del formulario conservan los límites observados de la plantilla salvo AGE: no equivalen a una validación universal de otras poblaciones.
