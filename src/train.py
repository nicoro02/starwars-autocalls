"""Entrenamiento y evaluacion del modelo de duracion.

Particion TEMPORAL, no aleatoria. El modelo se entrena con historia y se usa
para cotizar solicitudes futuras, asi que la evaluacion debe reproducir eso.

    2016-2022  ->  walk-forward para comparar modelos y decidir
    2023-2024  ->  bloque final, se toca una sola vez
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import RAIZ, preparar
from src.features import CATEGORICAS, FEATURES, NUMERICAS, TARGET, construir

MODELOS = RAIZ / "models"
ANYO_CORTE = 2023
SEMILLA = 42


def construir_modelo(nombre: str):
    if nombre == "media":
        # Baseline minimo. Un modelo que no lo supere no aprende nada.
        return DummyRegressor(strategy="mean")

    if nombre == "lineal":
        # Baseline de referencia, no candidato. El efecto de la volatilidad
        # cambia de signo segun basket_type y un lineal le asigna un unico
        # coeficiente, con lo que los dos efectos se cancelan.
        return Pipeline([
            ("prep", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            ], remainder="passthrough")),
            ("modelo", LinearRegression()),
        ])

    if nombre == "arbol":
        # Segmenta por basket_type y aprende el efecto correcto en cada rama.
        return HistGradientBoostingRegressor(
            categorical_features=CATEGORICAS, random_state=SEMILLA)

    raise ValueError(f"Modelo desconocido: {nombre}")


def ajustar_y_evaluar(nombre, usar_ratio, tr, te) -> dict:
    """Entrena con tr, predice sobre te y devuelve las metricas en meses."""
    y_tr = tr[TARGET] / tr["vida_nominal_meses"] if usar_ratio else tr[TARGET]

    modelo = construir_modelo(nombre)
    modelo.fit(tr[FEATURES], y_tr)
    pred = modelo.predict(te[FEATURES])

    if usar_ratio:
        pred = pred * te["vida_nominal_meses"]

    y_real = te[TARGET]
    return {
        "MAE": mean_absolute_error(y_real, pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, pred)),
        "R2": r2_score(y_real, pred),
    }


def walk_forward(df, nombre, usar_ratio, n_cortes=5) -> pd.DataFrame:
    """Bloques temporales crecientes: entrena con el pasado, evalua el futuro.

    test_size fija el bloque de evaluacion al 10%. Sin ese parametro
    TimeSeriesSplit reparte todo en n_cortes+1 bloques iguales y el primer
    corte entrenaria solo con un sexto de la muestra.
    """
    df = df.sort_values("requested_date").reset_index(drop=True)
    cortes = TimeSeriesSplit(n_splits=n_cortes, test_size=int(len(df) * 0.10))

    return pd.DataFrame([
        ajustar_y_evaluar(nombre, usar_ratio, df.iloc[tr], df.iloc[te])
        for tr, te in cortes.split(df)
    ])


def comparar(df) -> pd.DataFrame:
    filas = []
    for nombre in ["media", "lineal", "arbol"]:
        for usar_ratio in [False, True]:
            if nombre == "media" and usar_ratio:
                continue
            res = walk_forward(df, nombre, usar_ratio)
            filas.append({
                "modelo": nombre,
                "target": "ratio" if usar_ratio else "meses",
                "MAE_medio": res["MAE"].mean(),
                "MAE_sd": res["MAE"].std(),
                "RMSE_medio": res["RMSE"].mean(),
                "R2_medio": res["R2"].mean(),
            })
    return pd.DataFrame(filas).round(3)


def main():
    rfqs, volatilidad, referencia = preparar()
    df = construir(rfqs, volatilidad, referencia)

    anyo = df["requested_date"].dt.year
    desarrollo = df[anyo < ANYO_CORTE].copy()
    final = df[anyo >= ANYO_CORTE].copy()

    print(f"\ndesarrollo (2016-{ANYO_CORTE - 1}): {len(desarrollo)} RFQs")
    print(f"evaluacion final ({ANYO_CORTE}-2024): {len(final)} RFQs")

    print("\n=== Walk-forward sobre desarrollo, 5 cortes ===")
    tabla = comparar(desarrollo)
    print(tabla.to_string(index=False))

    mejor = tabla[tabla["modelo"] != "media"].sort_values("MAE_medio").iloc[0]
    nombre, usar_ratio = mejor["modelo"], mejor["target"] == "ratio"
    print(f"\nSeleccionado: {nombre} con target en {mejor['target']}")

    print("\nMAE por corte del modelo elegido:")
    print(walk_forward(desarrollo, nombre, usar_ratio).round(3).to_string())

    # Contraste, no validacion. Si el aleatorio saliera mucho mejor que el
    # temporal habria deriva y el modelo envejeceria rapido en produccion.
    tr, te = train_test_split(desarrollo, test_size=0.2, random_state=SEMILLA)
    ale = ajustar_y_evaluar(nombre, usar_ratio, tr, te)
    print(f"\nContraste con split aleatorio 80/20:")
    print(f"  MAE aleatorio {ale['MAE']:.3f}   MAE temporal {mejor['MAE_medio']:.3f}")

    print(f"\n=== Evaluacion final ({ANYO_CORTE}-2024, no usada para decidir) ===")
    res = ajustar_y_evaluar(nombre, usar_ratio, desarrollo, final)
    for k, v in res.items():
        print(f"  {k}: {v:.3f}")

    # El artefacto se entrena con todos los datos. Fijada la configuracion,
    # descartar los dos ultimos anyos solo empeoraria el modelo desplegado.
    y = df[TARGET] / df["vida_nominal_meses"] if usar_ratio else df[TARGET]
    modelo = construir_modelo(nombre)
    modelo.fit(df[FEATURES], y)

    MODELOS.mkdir(exist_ok=True)
    joblib.dump({
        "modelo": modelo,
        "features": FEATURES,
        "categoricas": CATEGORICAS,
        "numericas": NUMERICAS,
        "usar_ratio": usar_ratio,
        "metricas_test": res,
        "n_entrenamiento": len(df),
    }, MODELOS / "modelo.joblib")
    print(f"\nArtefacto guardado en {MODELOS / 'modelo.joblib'}")


if __name__ == "__main__":
    main()