# Adopción del experimento de Estefanía — RiskVision, Grupo 6

## Decisión del equipo

Se adopta formalmente como configuración candidata LightGBM original con paquete E, conservada por **Estefanía** en su experimento. Esto reconoce su contribución y evita repetir innecesariamente Optuna. **own_real_optimization_executed = False**: nuestra rama no ejecutó la búsqueda. La configuración sigue PROVISIONAL y pendiente de validación propia en 07.

Procedencia: origin/Estefania, commit e4067168e4f75dfe3c4f9f5e83cf20df797775c5. Fuentes: reports/06_best_params.json y reports/06_optimizacion.md; el texto de notebooks/06_optimizacion.ipynb aporta las cifras redondeadas de reserva. Experimento reportado: /Users/estefaniacg1995@gmail.com/riskvision-optimizacion. No se modifican ni se presentan como propios esos resultados.

## Configuración conservada

Paquete E: utilization_mean, pending_balance_last, delay_streak, n_zero_payments y log_limit, calculadas dentro del pipeline junto con los predictores originales sin ID ni objetivo.

| Parámetro | Valor |
|---|---:|
| n_estimators | 300 |
| learning_rate | 0.03 |
| num_leaves | 15 |
| min_child_samples | 50 |
| subsample | 0.8 |
| subsample_freq | 1 |
| colsample_bytree | 0.8 |
| random_state | 42 |
| verbose | -1 |
| n_jobs | 4 |

Los diez valores coinciden con candidates.py. decision.py mantiene una copia fija de la selección externa y rechaza discrepancias con los parámetros actuales, en lugar de reinterpretar silenciosamente la contribución de Estefanía.

## Evidencia reportada, sin reproducción propia

- 12 ensayos, objetivo ROC AUC maximizado; mejor AUC CV 0.7869, optimista por selección.
- AUC de reserva redondeado 0.7865 para original y optimizado, según el texto del notebook.
- Penalización 5:1 redondeada 0.549 original y 0.543 optimizado.
- IC del 95 % de penalización optimizado menos original: [-0.0185, 0.007166666666666667]. Cruza cero: su regla no respalda adoptar el optimizado. No demuestra equivalencia, superioridad del original ni ahorro.
- Penalización: falso negativo pesa 5 y falso positivo 1; umbral teórico 1/6. Es hipótesis económica del grupo, no costos observados del banco. No se adopta ese umbral como definitivo.

## Diferencias y límites

Su espacio busca reg_alpha; el nuestro busca max_depth y usa otros rangos y opciones de determinismo/hilos. Su decisión usa un IC de penalización; nuestro protocolo mantiene ROC AUC principal y separa bootstrap de AUC y penalización. No se cambia retrospectivamente un protocolo para atribuirle sus cifras.

La misma semilla y proporción de reserva no prueban igualdad de clientes sin huella, orden y versiones coincidentes. La reserva ya estuvo expuesta y la prueba original fue consultada: no son una evaluación completamente independiente. El notebook publicado no conserva salidas ejecutadas; los informes no contienen huella del entrenamiento, versiones completas ni run_id. Esos datos se pueden solicitar para fortalecer trazabilidad, sin volver a entrenar ni ejecutar Optuna.

Su experimento es evidencia externa adoptada por el equipo. El informe 07 debe contener solo métricas de su futura ejecución, con sus propias fechas, revisión y huella. El reconocimiento externo viaja en decision.antecedent; nunca se copia como internal_evaluation.
