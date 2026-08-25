"""Interpretacion del modelo: que variables importan y si tiene sentido de negocio.

Importancia por permutacion en lugar de la interna del modelo. Se calcula sobre
datos no vistos, asi que mide capacidad predictiva y no ajuste a la muestra, y
se expresa en la unidad de la metrica: cuanto empeora el MAE al destruir la
relacion de cada variable con el target.
"""

import pandas as pd
from sklearn.inspection import permutation_importance

from src.data import preparar
from src.features import FEATURES, TARGET, construir
from src.train import ANYO_CORTE, SEMILLA, construir_modelo

# Efecto esperado de cada variable, escrito a partir de la definicion del
# producto antes de entrenar. Contrastar el modelo contra esto es lo que
# permite decir si tiene sentido de negocio.
EXPECTATIVAS = {
    "vida_nominal_meses": "++  acota la duracion maxima posible",
    "n_oportunidades": "+   mas ocasiones de cancelar",
    "autocall_barrier_pct": "+   barrera mas alta, mas dificil de superar",
    "quoted_implied_vol": "-/+ negativo en single, positivo en worst_of",
    "basket_type": "+   worst_of dura mas, el peor se aleja de la barrera",
    "no_call_period_months": "+   impone un suelo a la duracion",
    "meses_entre_obs": "+   observar menos a menudo retrasa la cancelacion",
    "product_type": "+   plantilla de contrato, fija el resto de parametros",
    "n_subyacentes": "+   mas activos, peor es el peor de la cesta",
    "vol_realizada_max": "-/+ mismo mecanismo que la volatilidad implicita",
    "vol_estructural_max": "-/+ idem, nivel de largo plazo",
}


def ratio(df):
    return df[TARGET] / df["vida_nominal_meses"]


def importancia(modelo, df, n_repeticiones=10):
    res = permutation_importance(
        modelo, df[FEATURES], ratio(df),
        n_repeats=n_repeticiones, random_state=SEMILLA,
        scoring="neg_mean_absolute_error",
    )
    return (pd.DataFrame({"variable": FEATURES,
                          "importancia": res.importances_mean,
                          "sd": res.importances_std})
            .sort_values("importancia", ascending=False)
            .reset_index(drop=True))


def efecto_volatilidad(df, modelo):
    """Comprueba si el modelo aprendio el cambio de signo de la volatilidad.

    Quintiles de volatilidad implicita dentro de cada tipo de cesta. Si capto
    la interaccion, la fraccion de vida consumida debe decrecer en single y
    crecer en worst_of, tanto en el dato real como en la prediccion.
    """
    t = df[["basket_type", "quoted_implied_vol"]].copy()
    t["ratio_real"] = ratio(df)
    t["ratio_predicho"] = modelo.predict(df[FEATURES])
    t["quintil"] = t.groupby("basket_type")["quoted_implied_vol"].transform(
        lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5]))

    return (t.groupby(["basket_type", "quintil"], observed=True)
            .agg(vol_media=("quoted_implied_vol", "mean"),
                 ratio_real=("ratio_real", "mean"),
                 ratio_predicho=("ratio_predicho", "mean"),
                 n=("ratio_real", "size"))
            .round(3))


def main():
    rfqs, volatilidad, referencia = preparar(verbose=False)
    df = construir(rfqs, volatilidad, referencia)

    anyo = df["requested_date"].dt.year
    desarrollo = df[anyo < ANYO_CORTE].copy()
    evaluacion = df[anyo >= ANYO_CORTE].copy()

    # Se reentrena la configuracion elegida para inspeccionarla sobre datos
    # que el modelo no ha visto.
    modelo = construir_modelo("arbol")
    modelo.fit(desarrollo[FEATURES], ratio(desarrollo))

    print("=== Importancia por permutacion (caida de MAE al permutar) ===\n")
    imp = importancia(modelo, evaluacion)
    print(imp.round(4).to_string(index=False))

    print("\n=== Contraste con las expectativas de negocio ===\n")
    for _, f in imp.head(8).iterrows():
        esperado = EXPECTATIVAS.get(f["variable"], "sin expectativa previa")
        print(f"  {f['variable']:22s} {f['importancia']:.4f}   {esperado}")

    print("\n=== Efecto de la volatilidad por tipo de cesta ===")
    print("(fraccion de vida nominal consumida, por quintiles de vol)\n")
    print(efecto_volatilidad(evaluacion, modelo).to_string())


if __name__ == "__main__":
    main()