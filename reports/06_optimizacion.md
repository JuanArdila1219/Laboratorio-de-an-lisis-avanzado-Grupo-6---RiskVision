# 06 · Optimización con seguimiento

- **Finalista:** LightGBM con el paquete E (elegido en el notebook 05).
- **Métrica principal:** roc_auc (la misma del torneo). **Presupuesto:** 12 ensayos, respetado.
- **Mejor roc_auc en la búsqueda** (optimista, solo pliegues de búsqueda): 0.7869.
- **Mejores parámetros:** {'n_estimators': 200, 'learning_rate': 0.03822726574649208, 'num_leaves': 8, 'min_child_samples': 138, 'subsample': 0.7035119926400067, 'colsample_bytree': 0.831261142176991, 'reg_alpha': 6.388511557344611e-06, 'reg_lambda': 0.0004793052550782129}.
- **Importancia de los hiperparámetros:** colsample_bytree 0.40, num_leaves 0.28, subsample 0.13, learning_rate 0.09, min_child_samples 0.04, reg_lambda 0.03, n_estimators 0.02, reg_alpha 0.01
- **Comprobación honesta** en 6,000 clientes reservados que la búsqueda no vio: intervalo del 95 % de la diferencia de costo 5:1 por cliente [-0.0185, +0.0072].
- **Decisión** (regla escrita antes de mirar): parámetros de **partida**, porque el intervalo cruza el cero: la mejora no se distingue del ruido.
- **Seguimiento:** cada ensayo es una ejecución registrada (/Users/estefaniacg1995@gmail.com/riskvision-optimizacion, databricks (dentro del workspace)); umbral operativo pendiente para la etapa 7.
