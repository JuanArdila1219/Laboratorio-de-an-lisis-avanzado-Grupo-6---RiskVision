# 08 · Registro y persistencia de RiskVision

## 1. Estado y destino

Implementación preparada para pruebas sintéticas. Referencia: LightGBM original de candidates.py, paquete E, estado **PROVISIONAL**. No existe modelo definitivo aprobado ni evaluación real completa de 07. No se publicaron artefactos ni se creó versión o alias en servicios remotos. No se atribuyen métricas de Estefanía a esta implementación.

Rama de desarrollo: revision-h2-riskvision. La revisión exacta, fecha UTC y versiones efectivas se obtienen al generar cada paquete; no se inventa un commit para cambios todavía sin publicar. Experimento, nombre de Registry, versión remota y alias están pendientes de autorización.

## 2. Procedimiento y artefactos

1. Completar y revisar la evaluación 07. Congelar decisión, parámetros, calibración, esquema y política operativa con entrenamiento. La prueba original ya se consultó: no es una evaluación final completamente independiente y no debe seleccionar modelos o umbrales.
2. Mantener todos los controles desactivados hasta autorización. La demostración sintética es independiente del flujo real y utiliza únicamente un directorio temporal.
3. Guardar el pipeline completo ajustado con save_bundle. Nunca guardar solamente el estimador. Los transformadores y sus estadísticas aprendidas permanecen dentro del pipeline; no se vuelven a ajustar al cargar.
4. Crear una carpeta nueva con pipeline.joblib, input_schema.json, metadata.json y manifest.json. Guardar fuera de Git cualquier artefacto real y conservar externamente la huella del manifiesto. La función rechaza directorios existentes.
5. verify_bundle comprueba integridad sin deserializar. load_bundle exige origen confiable y verifica hashes antes de Joblib; para artefactos reales exige además la huella externa. Nunca cargar un pickle desconocido.
6. Comparar clases exactas y probabilidades (rtol=1e-14, atol=1e-15) mediante compare_predictions. Verificar también recuperación desde una sesión Python nueva.

Los metadatos contienen nombre/versión local, estado, parámetros efectivos, paquete E, procedencia de decisión, revisión del código, UTC, versiones, SHA-256, experimento y métricas de validación solo cuando se proporcionen. El envoltorio calibrado se conserva completo. La versión de Registry no se confunde con la local: se devuelve en un recibo separado, sin modificar ni invalidar el paquete original.

## 3. Contrato de entrada y continuidad hacia 09

Exactamente 23 predictores originales ya limpios, sin ID, objetivo ni variables derivadas externas. El pipeline calcula las variables E. Se conserva el orden del ajuste y se reordenan entradas equivalentes. Se aceptan faltantes para los imputadores, categorías válidas ausentes del ajuste (con advertencia) y facturas negativas plausibles. Se rechazan duplicados de columnas, infinitos, edades fraccionarias y dominios inválidos. Educación y matrimonio deben llegar con la agrupación fija existente; la persistencia no inventa otra limpieza.

09 podrá reutilizar load_bundle con trusted=True, expected_manifest_sha256 y require_approved=True. El objeto recuperado expone predict, predict_proba, classes_ y schema y es compatible con predict_with_scores. No se modificó 09. La clase del modelo y review_threshold del esquema auxiliar no equivalen a una política económica aprobada; la regla de negocio y su umbral necesitan revisión separada. La relación 5:1 continúa siendo una hipótesis del grupo, sin ahorro demostrado.

## 4. Controles y aprobación

Todos inicialmente False: ALLOW_FINAL_REAL_TRAINING, ALLOW_DEFINITIVE_REGISTRATION, ALLOW_MODEL_PROMOTION_OR_APPROVAL y ALLOW_REAL_ARTIFACT_PUBLICATION. El módulo aplica los controles, además de las condiciones del notebook. No basta con un comentario o activar una casilla.

El ajuste real reutiliza el entrenamiento de 07, exige configuración congelada y huella del entrenamiento. Registrar o promover como definitivo requiere estado approved_definitive, decisión congelada y aprobación externa con responsable, fecha e informe de validación. Este módulo **no concede** esa aprobación. Provisionales y sintéticos nunca pueden registrarse como definitivos. Los sintéticos solo se guardan bajo el temporal del sistema y no se publican remotamente, incluso si se activan controles.

## 5. Tracking frente a Model Registry

publish_bundle_artifacts crea un run nuevo para evidencia y los cuatro archivos, con FINISHED o FAILED según el resultado. Puede documentar un provisional real autorizado, sin crear una versión del modelo. register_definitive_model requiere aprobación y publica el flavor sklearn completo, firma, ejemplo explícito, código importable y dependencias; utiliza las probabilidades como salida pyfunc. La versión se toma directamente del registro de esa ejecución. No se busca la versión máxima ni se asigna champion automáticamente.

promote_model_version es una operación separada: verifica aprobación, recibo, versión exacta y huella remota antes de asignar un alias. Un fallo remoto puede dejar un run o una versión creados: revisar el recibo y MLflow antes de reintentar; no hay transacción distribuida ni recuperación automática. No se implementa aprobación criptográfica ni control de identidad: el registro de aprobación debe provenir de la revisión autorizada del grupo.

## 6. Evidencia y resultados pendientes

Las pruebas utilizan exclusivamente clientes sintéticos y mocks de servicios: round-trip, clases, probabilidades, transformadores aprendidos, hashes, alteraciones, esquema, carga confiable, versiones, controles, pipelines existentes calibrados, proceso Python nuevo y ejecución del notebook con todo desactivado. Comprobaciones locales del 6 de noviembre de 2026: 27 pruebas específicas aprobadas; regresión de 343 pruebas aprobada, excluyendo test_tuning.py y test_reporting.py para no ejecutar Optuna; una prueba adicional del protocolo de ajuste final aprobada mediante simulación. En total, 28 pruebas nuevas aprobadas. Se observó una advertencia de deprecación de SQLAlchemy en la prueba de Tracking local, sin fallos. JSON del notebook válido, estructura de las 15 celdas y metadatos conservados. Estos resultados no son métricas de validación real.

Pendientes: evaluación real 07, aprobación, entrenamiento final autorizado, paquete real, comparación real de ida y vuelta, firma efectiva en Databricks, permisos y registro remoto. Ningún resultado de estas operaciones se ha fabricado.

## 7. Preparación para Databricks y limitaciones

Reutilizar requirements-serverless-06.txt desde Environment, reiniciar la sesión y ejecutar desde la primera celda. No se requieren nuevas bibliotecas. Configurar RISKVISION_PROJECT_ROOT, trazabilidad de rama/commit cuando Git no esté disponible, experimento completo y destino persistente autorizado (por ejemplo un Volume), fuera del checkout y de Git. Reproducir las versiones registradas del artefacto; el cargador exige las bibliotecas principales exactas y Python de la misma versión mayor/menor. El entorno base y dependencias transitivas no constituyen un lock completo.

Para Unity Catalog indicar catalog.schema.model y comprobar permisos USE CATALOG, USE SCHEMA y CREATE MODEL, además de acceso al experimento/almacenamiento. Verificar firmas, imports empaquetados y recuperación en una sesión nueva antes de registrar. La integración de Registry se probó mediante simulaciones locales, no mediante una ejecución remota en Serverless. SHA-256 detecta corrupción, pero no sustituye confianza en la fuente ni una firma digital. La clasificación synthetic/real es una declaración explícita del llamador, no una prueba automática de procedencia de los datos.

## Traspaso formal 07→08→09 y responsabilidad del ajuste final

08 es el único notebook que ajusta con los 24.000 clientes. Recibe un contrato JSON nuevo y su huella externa desde 07, confirmando procedencia. read_handoff comprueba autoría de la evidencia externa, configuración congelada, informe propio, versiones, orden de 23 predictores, SHA-256 del entrenamiento y autorización humana del ajuste final. No se toman las cifras de Estefanía como métricas de validación propia.

MODEL_DECISION puede seguir provisional: su nombre no implica aprobación. APPROVE_DEFINITIVE_DECISION=False y DEFINITIVE_APPROVAL_RECORD=None impiden aprobar automáticamente. approve_for_inference requiere aprobación externa explícita con responsable, UTC, identificador de informe y huella de configuración. La creación del contrato, el entrenamiento, el guardado, el registro y la promoción no conceden aprobación.

Con ALLOW_FINAL_REAL_TRAINING autorizado se carga únicamente train.joblib con huella y dimensiones verificadas; se comprueban versiones y orden y se aplica la calibración congelada con el mecanismo existente. No hay ajuste final en 07 ni en 09. El paquete verifica correspondencia entre calibración real, contrato y esquema; las métricas guardadas se toman del candidato raw/sigmoid seleccionado del informe 07 y llevan prefijo internal_reserve_, no son métricas del modelo reajustado con 24.000 clientes ni de prueba independiente.

Persistir datos reales requiere ALLOW_REAL_ARTIFACT_PUBLICATION: en la orquestación actual también habilita publicación de artefactos en Tracking, por lo que se debe autorizar el destino y el experimento conjuntamente. Registrar y asignar alias continúan siendo permisos independientes. Las constancias reflejan la decisión efectiva. La lectura de paquetes antiguos conserva compatibilidad; los nuevos ajustes finales requieren el contrato. No se ejecutó ninguna operación real en esta integración.

El registro externo de aprobación definitiva debe contener approved=true, approved_by, approved_at en UTC, validation_report_id y configuration_sha256 iguales a los del contrato revisado. Ninguno se rellena automáticamente en el notebook. Para reproducir el código, usar la misma revisión aprobada del proyecto en las tres etapas; la huella del artefacto no autentica por sí sola el código fuente que se importa.
