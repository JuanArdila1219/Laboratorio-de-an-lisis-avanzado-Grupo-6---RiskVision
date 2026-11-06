# 07 · Validación e interpretación — RiskVision, Grupo 6

## Estado

Implementación preparada y comprobable con datos sintéticos. No hay resultados reales propios de etapa 07. Los cuatro permisos de ajuste, calibración, evaluación complementaria y exportación definitiva comienzan desactivados. No existe todavía un modelo definitivo aprobado.

## Modelo provisional y procedencia

- LightGBM con paquete E: utilization_mean, pending_balance_last, delay_streak, n_zero_payments y log_limit, además de las entradas originales sin ID ni objetivo.
- Parámetros originales de `src/proyecto/models/candidates.py:LGBM_PARAMS`, sin cambios. Cada ejecución registra su copia completa, rama, commit, versiones, fecha UTC y SHA-256 del entrenamiento.
- Regresión logística con E: referencia explicable con sus parámetros originales, escalado y PAY_* como categorías en su pipeline.
- Optimizado: no disponible por defecto. Requiere evidencia completa, configuración/espacio/semillas compatibles, huella de entrenamiento coincidente y aprobación. El módulo valida sin lanzar Optuna.
- Antecedente: origin/Estefania, commit e4067168e4f75dfe3c4f9f5e83cf20df797775c5, conservó parámetros originales en otro experimento con otra búsqueda y regla de comparación. No se atribuyen sus cifras a esta implementación. Nuestra optimización real continúa pendiente.

## Protocolo propuesto para revisión antes de ejecutar

1. Leer exclusivamente el entrenamiento aprobado (24.000 × 25), verificar huella, ID, objetivo y esquema. No rehacer limpieza ni partición original.
2. Separar 75/25 estratificado, semilla 7: 18.000 para desarrollo y 6.000 de reserva interna. Los clientes de reserva ya participaron indirectamente en etapas anteriores; no son una evaluación independiente.
3. En desarrollo, obtener una predicción OOF por cliente mediante cinco pliegues estratificados, semilla 42. Cada pipeline se clona y ajusta solo con su entrenamiento. No contar predicciones de repeticiones como clientes independientes.
4. Si se autoriza calibración, comparar raw/sigmoid. El calibrador se ajusta con CV de tres pliegues dentro del entrenamiento de cada pliegue OOF, con ensemble=False. No aprende de la reserva ni de la prueba.
5. Recomendación descriptiva de calibración: Brier y log loss menores, sin descenso del AUC OOF. No implica selección definitiva ni significación estadística.
6. Proponer umbral usando únicamente OOF de desarrollo: grilla 0,05–0,95 en pasos de 0,05 más 1/6; minimizar penalización hipotética 5:1, desempatar por menor carga y mayor umbral. No optimiza AUC, que no depende del umbral. El umbral requiere aprobación posterior; 1/6 no es óptimo garantizado sin calibración y matriz de costos válida.
7. Ajustar candidatos con desarrollo y comprobarlos en la misma reserva. ROC AUC es principal; penalización, calibración y carga son controles. No promover automáticamente un ganador.
8. Congelar expresamente calibración y umbral por candidato tras revisar entrenamiento. Con permiso, ajustar los pipelines para evaluación con los 24.000 clientes. Esto sigue siendo una decisión provisional, no aprobación definitiva.
9. Solo con autorización separada, evaluar la prueba previamente expuesta como comprobación complementaria. No seleccionar ni corregir el modelo, calibración o umbral con ella.

## Métricas previstas y resultados pendientes

| Área | Métricas/preparación | Estado real |
|---|---|---|
| Discriminación | ROC AUC principal; Average Precision | Pendiente |
| Decisión | Precision, recall y F1 clase 1; matriz TN/FP/FN/TP | Pendiente |
| Probabilidades | Brier, log loss, curvas OOF, ECE | Pendiente |
| Negocio | Penalización relativa 5:1, proporción marcada, recall al revisar 20 % | Pendiente |
| Comparación | Bootstrap pareado AUC; bootstrap de penalización separado | Pendiente |
| Segmentos | SEX y tramos AGE [18,30), [30,45), [45,60), [60,101) | Pendiente |
| Interpretación | Permutación del pipeline completo, cinco repeticiones, semilla 42 | Pendiente |

No hay métricas numéricas reales en este documento. Segmentos con una sola clase informan AUC/AP no disponibles. La capacidad del 20 % es supuesta. La penalización 5:1 no equivale a dinero ahorrado.

## Intervalos y límites de inferencia

Se remuestrean los mismos clientes para ambos modelos, 1.000 veces, semilla 42. Para AUC se recalculan los dos AUC en cada remuestra, se omiten las de una clase y se informa el número válido. Variante − original positivo favorece a la variante.

El bootstrap separado de penalización recalcula ambos costos relativos medios; variante − original negativo favorece a la variante. Los intervalos percentiles del 95 % son condicionales a los modelos ajustados. No incluyen la incertidumbre de repetir ajuste/selección ni eliminan la exposición previa.

La prueba original ya fue evaluada y el EDA exploró el conjunto completo. No se crea una marca ficticia de primera apertura ni se presenta la evaluación complementaria como una prueba final independiente. Una evaluación externa independiente requeriría nuevos datos no utilizados para decidir.

## Interpretación y segmentos

Las importancias por permutación corresponden a las entradas originales y a sus efectos a través del paquete E, no a contribuciones separadas de cada derivada. Correlación entre variables puede repartir o distorsionar importancias. No son efectos causales ni criterios para negar crédito. SHAP es opcional y no se incorpora hasta comprobar utilidad y compatibilidad.

El análisis por SEX y edad es descriptivo, con tamaño y frecuencia de cada grupo. No justifica automáticamente intervenciones distintas por atributos sensibles. No se retocan variables después de consultar la prueba.

## Trazabilidad, guardado y pasos hacia 08/09

El notebook prepara un registro de decisión y un informe con código, versiones, huella de datos, parámetros, permisos y resultados disponibles. Tras una evaluación autorizada, MLflow recibe un informe provisional en un run nuevo de etapa 07; no se sobrescriben informes ni se registran modelos como definitivos.

La exportación definitiva requiere un permiso independiente, decisiones congeladas y status=approved_definitive, que este registro provisional no establece. La escritura del artefacto es exclusiva, sin sobrescribir modelos previos. La entrega a 08/09 requiere revisar y aprobar configuración, resultados, limitaciones y pruebas de guardar/cargar/predecir en el entorno de destino.

## Decisiones pendientes de autorización

- Aprobar o ajustar el protocolo antes de entrenar con datos reales.
- Autorizar ajuste y, separadamente, calibración.
- Revisar comparación y elegir configuración operativa por validación.
- Autorizar la evaluación complementaria y conservar su advertencia de exposición.
- Aprobar, en otra fase, un modelo definitivo y su exportación/registro.

## Entorno

`requirements-serverless-07.txt` hereda las versiones verificadas de etapa 06 y propone matplotlib 3.11.2 para curvas (verificado localmente, pendiente en Serverless). No instala SHAP. Seleccionar el entorno base Python 3.12.3 y configurar la dependencia desde Environment antes de una sesión nueva. La primera celda verifica rutas, imports y dependencias antes de deserializar datos; no usa parches de pandas ni celdas temporales de pruebas anteriores.
