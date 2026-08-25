"""Integracion de las tres tablas e ingenieria de features.

Las tres fuentes tienen grano distinto: una fila por solicitud, una por
(subyacente, dia) y una por subyacente. El grano de trabajo es la RFQ, asi que
se explota la cesta, se cruza con las dos tablas de mercado y se reagrega.
"""

import pandas as pd

# Descartadas en la exploracion: protection_barrier_pct (correlacion espuria),
# notional_credits, counterparty y trader_id (sin senyal util), y
# requested_date, que se usa para el split y el cruce pero no como feature.

CATEGORICAS = ["product_type", "basket_type"]

NUMERICAS = [
    "autocall_barrier_pct",
    "no_call_period_months",
    "quoted_implied_vol",
    "meses_entre_obs",
    "vida_nominal_meses",
    "n_oportunidades",
    "n_subyacentes",
    "vol_realizada_max",
    "vol_realizada_mean",
    "vol_realizada_min",
    "vol_realizada_std",
    "vol_estructural_max",
    "vol_estructural_mean",
    "vol_estructural_min",
    "vol_estructural_std",
]

FEATURES = CATEGORICAS + NUMERICAS
TARGET = "avg_duration_months"


def explotar_cesta(rfqs: pd.DataFrame) -> pd.DataFrame:
    """Convierte 'CLNE|DRC' en una fila por subyacente."""
    return (
        rfqs[["rfq_id", "requested_date", "underlyings"]]
        .assign(underlying=lambda d: d["underlyings"].str.split("|"))
        .explode("underlying")
        .drop(columns="underlyings")
        .reset_index(drop=True)
    )


def unir_volatilidad_pit(cesta: pd.DataFrame, volatilidad: pd.DataFrame) -> pd.DataFrame:
    """Cruce point-in-time con la volatilidad realizada.

    La tabla de mercado llega hasta 2026 y la ultima RFQ es de 2024, asi que un
    merge sin restriccion temporal meteria volatilidad futura como predictor.
    direction='backward' toma la ultima observacion en o antes de la fecha.
    """
    cesta = cesta.sort_values("requested_date")
    volatilidad = volatilidad.sort_values("date")

    unido = pd.merge_asof(
        cesta,
        volatilidad,
        left_on="requested_date",
        right_on="date",
        by="underlying",
        direction="backward",
    )

    if not (unido["date"] <= unido["requested_date"]).all():
        raise ValueError("FUGA TEMPORAL: volatilidad posterior a la fecha de la RFQ")
    if unido["realized_vol_63d"].isna().any():
        raise ValueError("Hay subyacentes sin volatilidad previa a la RFQ")

    return unido


def agregar_cesta(cesta: pd.DataFrame) -> pd.DataFrame:
    """Colapsa los subyacentes de cada cesta en una fila.

    En un worst_of la cancelacion depende del activo mas debil, no del
    promedio, asi que la media sola pierde informacion. El max identifica al
    candidato a peor de la cesta y la std mide el riesgo de que uno se
    descuelgue y arrastre al conjunto.
    """
    agregado = cesta.groupby("rfq_id").agg(
        vol_realizada_max=("realized_vol_63d", "max"),
        vol_realizada_mean=("realized_vol_63d", "mean"),
        vol_realizada_min=("realized_vol_63d", "min"),
        vol_realizada_std=("realized_vol_63d", "std"),
        vol_estructural_max=("structural_base_vol", "max"),
        vol_estructural_mean=("structural_base_vol", "mean"),
        vol_estructural_min=("structural_base_vol", "min"),
        vol_estructural_std=("structural_base_vol", "std"),
    )

    # En cestas de un solo activo la desviacion no esta definida. Cero es el
    # valor correcto, no hay dispersion posible entre un unico elemento.
    for columna in ["vol_realizada_std", "vol_estructural_std"]:
        agregado[columna] = agregado[columna].fillna(0.0)

    return agregado.reset_index()


def derivar_features(rfqs: pd.DataFrame) -> pd.DataFrame:
    df = rfqs.copy()

    # Lo que mueve la duracion no es cada cuanto se observa, sino cuantas veces
    # puede cancelarse el producto. El no-call se descuenta porque durante esos
    # meses las observaciones no pueden activar la cancelacion.
    df["n_oportunidades"] = (
        (df["vida_nominal_meses"] - df["no_call_period_months"]) / df["meses_entre_obs"]
    )

    return df


def construir(rfqs: pd.DataFrame, volatilidad: pd.DataFrame,
              referencia: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de integracion. Devuelve una fila por RFQ."""
    cesta = explotar_cesta(rfqs)
    cesta = unir_volatilidad_pit(cesta, volatilidad)
    cesta = cesta.merge(referencia, on="underlying", how="left", validate="many_to_one")

    if cesta[["sector", "structural_base_vol"]].isna().any().any():
        raise ValueError("Hay subyacentes sin fila en la tabla de referencia")

    df = rfqs.merge(agregar_cesta(cesta), on="rfq_id", how="inner", validate="one_to_one")

    return derivar_features(df)


if __name__ == "__main__":
    from src.data import preparar

    rfqs, volatilidad, referencia = preparar()
    df = construir(rfqs, volatilidad, referencia)
    print(f"\nfilas: {len(df)}   features: {len(FEATURES)}\n")
    print(df[NUMERICAS].describe().T.round(3).to_string())