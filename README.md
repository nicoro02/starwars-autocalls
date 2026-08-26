# Estimación de duración de autocallables

Modelo que predice `avg_duration_months`, la duración media de un producto
estructurado hasta su cancelación anticipada o vencimiento.

La mesa necesita ese número al cotizar, cuando el producto todavía no existe.
Como el target viene de una simulación, el modelo es en realidad una
aproximación rápida de un Monte Carlo caro. No estima un fenómeno ruidoso,
sustituye un cálculo que ya existe pero es lento.

## Resultado

Bloque 2023-2024, reservado y no usado para ninguna decisión de modelado.

| MAE | RMSE | R² |
|---|---|---|
| **4,01 meses** | 5,46 | 0,941 |

Sobre productos que duran 40 meses de media, un error en torno al 10%.

Comparación en desarrollo (2016-2022, walk-forward de 5 cortes):

| Modelo | Target | MAE | RMSE | R² |
|---|---|---|---|---|
| Media global | meses | 16,92 | 21,40 | -0,00 |
| Lineal | meses | 9,90 | 12,73 | 0,65 |
| Lineal | ratio | 8,43 | 11,45 | 0,71 |
| Gradient boosting | meses | 3,93 | 5,42 | 0,94 |
| **Gradient boosting** | **ratio** | **3,81** | **5,25** | **0,94** |

## Uso

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Entrenar desde los CSV de origen. Ejecuta todo el pipeline y deja el artefacto
en `models/modelo.joblib`:

```bash
python -m src.train
```

Levantar la API. Carga el artefacto ya entrenado, sin reentrenar:

```bash
uvicorn src.api:app --reload
```

Documentación interactiva en `http://127.0.0.1:8000/docs`.

Importancia de variables y contraste con las expectativas de negocio:

```bash
python -m src.interpretar
```

Tests:

```bash
pytest -v
```

### Ejemplo

Enviando los términos de una RFQ a `/predict`:

```json
{
  "product_type": "Kessel Run Snowball",
  "underlyings": "CLNE|DRC",
  "basket_type": "worst_of",
  "autocall_barrier_pct": 1.0,
  "no_call_period_months": 3,
  "observation_frequency": "Monthly",
  "quoted_implied_vol": 0.2455,
  "requested_date": "2023-09-08",
  "start_date": "2023-09-08",
  "end_date": "2028-09-08"
}
```

```json
{
  "duracion_media_meses": 27.39,
  "vida_nominal_meses": 60,
  "fraccion_vida_consumida": 0.4565
}
```

La API recibe los términos del contrato, no las features del modelo. Quien
cotiza tiene el contrato delante, no la volatilidad agregada de la cesta. Las
features se construyen dentro importando las mismas funciones del pipeline, de
forma que entrenamiento e inferencia no puedan divergir.

## Estructura

```
data/         los tres CSV de origen
models/       artefacto entrenado
notebooks/    01_exploracion, análisis y decisiones
src/
  data.py         carga y limpieza
  features.py     integración de las tres tablas e ingeniería de features
  train.py        comparación de modelos, validación temporal y artefacto
  interpretar.py  importancia de variables
  api.py          API de inferencia
tests/        tests del pipeline
```

`data/` y `models/` se versionan a propósito, en contra de lo habitual, para
que el repositorio sea reproducible sin dependencias externas y se pueda probar
la inferencia sin reentrenar.

## Decisiones

El detalle está en `notebooks/01_exploracion`. Lo esencial:

| Hallazgo | Decisión | Por qué |
|---|---|---|
| 11.204 filas (45%) sin target | Fuera del entrenamiento, se conservan para analizar sesgo | Nulos estructurales, si el producto no se emitió no hay duración |
| 18 etiquetas para 6 frecuencias reales | Unificar y pasar a meses entre observaciones | Mismo concepto en inglés, español, abreviatura y forma larga |
| La frecuencia como categoría pierde la magnitud | Derivar `n_oportunidades` | Importa cuántas veces puede cancelarse el producto, no cada cuánto se observa |
| 303 filas (2,2%) con duración > vida nominal | Eliminar y reportar | Imposible por definición, y concentrado en un producto (9,2%) |
| `protection_barrier_pct` correlaciona -0,32 | Descartar | Desaparece al condicionar por `product_type`, es un proxy suyo (paradoja de Simpson) |
| `counterparty` y `trader_id` | Descartar | p=0,674 y eta2=0,43%. Además dependerían de categorías nuevas en producción |
| La volatilidad cambia de signo según la cesta | Conservar `basket_type` pese a la redundancia | Codifica la interacción en una sola partición binaria |
| El mercado llega dos años más allá de la última RFQ | Cruce point-in-time con `merge_asof` | Un merge sin restricción temporal metería volatilidad futura |
| Cestas de 1 a 3 activos | Agregar con max, min, media y desviación | En un `worst_of` manda el más débil, no el promedio |

## Modelado

**Validación temporal.** El modelo se entrena con historia y cotiza solicitudes
futuras, así que la evaluación tiene que reproducir eso. Con un split aleatorio
entrenaría con RFQs posteriores a las que evalúa.

- 2016-2022, walk-forward de 5 cortes, para comparar modelos y decidir
- 2023-2024, bloque reservado, se toca una sola vez

Hice también el split aleatorio, no como validación sino como contraste: MAE
3,72 frente a 3,81 del temporal. Y mirando el MAE de cada corte (4,18 · 4,44 ·
3,66 · 2,73 · 4,05) no se ve tendencia: el modelo ni mejora al acumular
historia ni se degrada con el tiempo. Las dos cosas indican que la relación es
estable y que no envejecerá rápido.

**Por qué árboles y no un lineal.** El efecto de la volatilidad sobre la
duración es negativo en cestas `single` (-0,51) y positivo en `worst_of`
(+0,20). Con un solo subyacente, más volatilidad hace más probable superar la
barrera y cancelar antes. En un `worst_of` se mira el peor activo de la cesta,
y más dispersión hunde al mínimo alejándolo de la barrera superior. Un lineal
asigna un único coeficiente a esa variable y los dos efectos se cancelan. Un
árbol segmenta por `basket_type` y aprende el efecto correcto en cada rama. La
tabla de resultados lo confirma, el error cae a la mitad.

**Target normalizado.** El modelo aprende la fracción de vida nominal que
consume el producto en vez de los meses absolutos, y la predicción se devuelve
siempre en meses. La escala del plazo ya viene dada por el contrato y el ratio
está acotado en [0, 1].

## Qué variables importan

Importancia por permutación sobre el bloque de evaluación, medida como cuánto
empeora el MAE al destruir cada variable.

| Variable | Importancia |
|---|---|
| `product_type` | 0,083 |
| `n_oportunidades` | 0,047 |
| `vol_estructural_max` | 0,041 |
| `autocall_barrier_pct` | 0,033 |
| `vol_estructural_std` | 0,016 |
| `vida_nominal_meses` | 0,014 |

Contrastado con las expectativas que escribí antes de entrenar, el modelo se
comporta como predice la lógica del producto. La comprobación más directa es el
efecto de la volatilidad por tipo de cesta, sobre datos no vistos:

| Cesta | Vol baja a alta | Fracción de vida consumida |
|---|---|---|
| `single` | 0,135 a 0,430 | 0,607 · 0,381 · 0,374 · 0,364 · 0,232 |
| `worst_of` | 0,184 a 0,385 | 0,611 · 0,617 · 0,608 · 0,691 · 0,747 |

Decreciente en un caso y creciente en el otro, y las predicciones siguen las
dos tendencias. El modelo captó la interacción sin que se la especificara.

## Limitaciones

**`vol_estructural_max` funciona como identificador encubierto.** Solo hay 14
subyacentes, cada uno con un valor fijo y distinto, así que la variable
identifica qué activos hay en la cesta más que medir su volatilidad. El modelo
ha memorizado el comportamiento de cada uno. Con un activo nuevo tendrá que
apoyarse en la volatilidad realizada, que ahora infrautiliza, y funcionará peor.

**La importancia por permutación subestima variables correlacionadas.**
`quoted_implied_vol` sale con importancia nula aunque su efecto es real y el
modelo lo captura. Al destruirla, se apoya en las demás variables de
volatilidad. Importancia cero significa prescindible dado el resto, no
irrelevante.

**El sesgo de selección solo se puede descartar sobre lo observable.** Las RFQs
ejecutadas y no ejecutadas son indistinguibles en todas las variables
disponibles. Si ejecutarse dependiera del precio cotizado o de la negociación
con el cliente, que no están en estas tablas, el sesgo existiría y no lo vería.

**`structural_base_vol` es leakage potencial de baja severidad.** Es una
estimación estática sin fecha, así que si se calibró con el histórico completo
incorpora información posterior a las RFQs antiguas. No verificable con estos
datos.

**El modelo no extrapola.** Al ser de árboles no predice fuera del rango visto
en entrenamiento. Un producto con términos atípicos, plazos de más de 120 meses
o barreras fuera de 0,95-1,40, recibiría la predicción del extremo más cercano.

**Sin optimización de hiperparámetros.** Valores por defecto de
`HistGradientBoostingRegressor`. Dado el alcance del ejercicio prioricé
justificar las decisiones sobre exprimir la métrica. Es la siguiente mejora
natural.

**El modelo parece haber saturado.** El MAE por corte no mejora aunque el
conjunto de entrenamiento crezca del 50% al 90% del histórico. Más datos del
mismo tipo no van a mejorarlo; para eso harían falta features nuevas.

## Para el equipo de datos

Dos cosas que exceden la limpieza y convendría corregir en origen.

**Wretched Hive Digital** concentra el 70% de las duraciones imposibles, con un
9,2% de sus filas frente a menos del 2,2% en el resto. Apunta a un fallo en la
simulación de ese producto.

**`structural_base_vol` no describe bien a REBL ni a HTTX.** Doce de los
catorce subyacentes mantienen un ratio estable entre volatilidad realizada y
estructural, entre 1,20 y 1,25. REBL se va a 2,66 y HTTX a 1,75, y en los dos
casos la volatilidad alta es persistente durante los trece años de histórico,
no episodios puntuales que inflen la media.