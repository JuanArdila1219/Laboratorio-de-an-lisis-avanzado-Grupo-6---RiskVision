# Especificación · Etapa 3 · Preprocesamiento (limpieza y partición)

> Se llena **antes** de pedirle código a Claude, a partir de la tabla de decisiones de
> `especificaciones/02-eda.md`. Claude la lee y la implementa; ustedes revisan que el código haga
> **exactamente** esto (`material-del-curso/kit/checklist-revision-ia.md`).
> Guía: `material-del-curso/kit/guia-eda-y-preprocesamiento.md`, sección 4 · Ejemplo: `material-del-curso/ejemplo-creditsight/src/creditsight/data/`.

**Grupo:** 6 (RiskVision)   **Fecha:** 2026-10-30   **Estado:** ☐ borrador  ☒ cerrada

## 1. Unidad de análisis y partición

- **Una fila es:** un cliente de tarjeta de crédito (30.000 clientes distintos).
- **Entidad que se repite (columna):** `ID`  ☒ no se repite → partir por fila es válido porque: los 30.000 `ID` son únicos, así que ningún cliente puede quedar en entrenamiento y en prueba. `split_stratified` lo verifica en cada ejecución.
- **Se parte por:** ☐ grupo (`GroupShuffleSplit` / `StratifiedGroupKFold`) ☒ fila, estratificada por `default payment next month` (22,12 % incumple)
- **Proporción de prueba:** 20 %   **Semilla:** 42 (24.000 entrenamiento, 6.000 prueba; igual a la del notebook de selección de modelo)
- **¿Hay orden temporal que respetar?** ☒ no ☐ sí → cada fila es un cliente con su historial de 6 meses ya resumido en columnas; no hay serie de tiempo por cliente.
- **Dónde se guarda la prueba:** `data/processed/` (cerrada: **no se usa para decidir** hasta la etapa 7; el archivo sí se sube al repositorio, decisión del 2026-11-05)

## 2. Datos personales

| Columna | Qué es | Se elimina en | Prueba que lo verifica |
|---|---|---|---|
| Ninguna | El dataset solo trae un `ID` anónimo (sin nombres, documentos ni contactos). `ID` se conserva para verificar la partición y se excluye de los predictores con `make_xy` | `make_xy()` (al modelar) | `test_make_xy_drops_the_id_and_the_target_from_the_predictors` |

## 3. Reglas de limpieza

Una fila por decisión del EDA. **Una función, una prueba.** Se marca como vacío lo imposible; no se borran filas sin criterio.

| # (de la tabla del EDA) | Columna(s) | Problema | Regla exacta (con el umbral o valor) | Función en `src/proyecto/data/cleaning.py` | Prueba en `tests/test_cleaning.py` |
|---|---|---|---|---|---|
| 1 | `EDUCATION` | Códigos 0, 5 y 6 sin documentar (345 clientes, 1,15 %) | 0, 5 y 6 pasan a 4 («otros») | `group_rare_education` | `test_education_rare_codes_become_others_and_normal_stay` |
| 2 | `MARRIAGE` | Código 0 sin documentar (54 clientes) | 0 pasa a 3 («otros») | `group_rare_marriage` | `test_marriage_zero_becomes_others_and_normal_stay` |
| 3 | `SEX`, `EDUCATION`, `MARRIAGE` | Son etiquetas guardadas como números | Se marcan como categorías | `to_category` | `test_to_category_marks_label_columns_only` |
| 4 | `AGE`, `SEX`, `LIMIT_BAL`, `PAY_AMT*` | Valores imposibles (no hay casos hoy: edad 21-79, límite 10.000-1.000.000, ningún pago negativo) | `AGE` fuera de 18-100, `SEX` distinto de 1 o 2, `LIMIT_BAL` ≤ 0 o `PAY_AMT*` < 0 pasan a vacío | `mark_impossible_as_missing` | `test_impossible_values_become_missing_and_normal_ones_stay` |
| 5 | Todas | No borrar filas; `BILL_AMT` negativo es saldo a favor | `clean` devuelve las mismas filas y conserva los negativos | `clean` | `test_clean_keeps_every_row_and_does_not_touch_the_original`, `test_negative_bill_is_kept` |

## 4. Vacíos

| Columna | Qué significa el vacío | Tratamiento hoy | Tratamiento en la etapa 4 (dentro del `Pipeline`) |
|---|---|---|---|
| Todas | ☒ no aplica: el conjunto no tiene vacíos (0 de 30.000 × 25) | ☒ dejar vacío (lo imposible se marca como vacío) | La mediana para los números y la moda para las categorías se aprenden del entrenamiento dentro del `Pipeline`, como medida de seguridad con datos nuevos |

> Imputar con medias o medianas **no se hace aquí**: depende del entrenamiento y va dentro del `Pipeline`.

## 5. Entradas y salidas

| | Ruta |
|---|---|
| Entrada | `data/raw/default of credit card clients.xls` (nunca se modifica) |
| Salida: entrenamiento | `data/processed/train.joblib` |
| Salida: prueba (cerrada) | `data/processed/test.joblib` |
| Código de limpieza | `src/proyecto/data/cleaning.py` |
| Código de partición | `src/proyecto/data/split.py` |
| Notebook que orquesta | `notebooks/03_preprocesamiento.ipynb` (solo llama a las funciones) |

## 6. Pruebas que deben existir

- ☒ **Partición:** ninguna entidad aparece en dos particiones y no se pierde ninguna fila (`tests/test_split.py`).
- ☒ **La prueba de partición falla** si se cambia temporalmente la partición. Como no hay entidades repetidas, se verificó quitando `stratify`: `test_split_keeps_the_default_rate_in_both_partitions` falla, y al restaurarlo vuelve a pasar.
- ☒ **Una prueba por regla de limpieza** de la sección 3, con un caso normal y un caso sucio hechos a mano.
- ☒ **Datos personales:** no hay columnas personales; `ID` se excluye de los predictores con `make_xy`.
- ☒ `pytest` en verde desde la raíz del repo.

## 7. Criterios de cierre (H1)

- ☒ La limpieza corre **de principio a fin desde cero** (`data/processed/` borrada y regenerada con el notebook 03), sin pasos manuales.
- ☒ Unidad de análisis y forma de partir **escritas** (sección 1) y documentadas en `reports/01_eda.md`.
- ☒ `pytest` en verde.
- ☒ Conjunto de prueba guardado y sin mirar.
- ☒ `data/raw/` intacta y `.env` fuera de Git (`git status`).
- ☐ Todo subido a GitHub.
- ☐ Revisaron el código de Claude con `checklist-revision-ia.md` y cada integrante puede explicar una función.
- ☐ Todo subido a GitHub y revisión del código con `checklist-revision-ia.md` (pendiente del grupo).
- ☒ **Hallazgo que no esperaban** (para la puesta en común): la regla simple «`PAY_0` ≥ 2» cuesta casi lo mismo que marcar a todos los clientes (0,775 frente a 0,779), y `n_atraso` (meses con atraso) se relaciona más con el objetivo (0,40) que `PAY_0` (0,33).
