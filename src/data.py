"""Carga y limpieza de las tres tablas de origen."""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"

# Las 18 etiquetas de observation_frequency son 6 frecuencias reales escritas
# de formas distintas. Se mapean a meses entre observaciones en lugar de
# tratarlas como categorias, que perderia el orden y la magnitud.
MESES_ENTRE_OBSERVACIONES = {
    "1D": 1 / 21,                                                    # ~21 dias habiles al mes
    "1M": 1.0, "M": 1.0, "Monthly": 1.0, "mensual": 1.0, "1 month": 1.0,
    "2M": 2.0,
    "3M": 3.0, "Q": 3.0, "Quarterly": 3.0, "trimestral": 3.0, "3 months": 3.0,
    "6M": 6.0,
    "1Y": 12.0, "Y": 12.0, "12M": 12.0, "Annual": 12.0, "anual": 12.0,
}

COLUMNAS_FECHA = ["requested_date", "start_date", "end_date"]


def cargar(ruta: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rfqs = pd.read_csv(ruta / "rfqs.csv")
    volatilidad = pd.read_csv(ruta / "daily_volatility.csv")
    referencia = pd.read_csv(ruta / "underlyings_reference.csv")
    return rfqs, volatilidad, referencia


def normalizar_frecuencia(serie: pd.Series) -> pd.Series:
    """Traduce las etiquetas de frecuencia a meses entre observaciones.

    Falla ante una etiqueta desconocida en vez de generar un nulo silencioso.
    """
    desconocidas = set(serie.unique()) - set(MESES_ENTRE_OBSERVACIONES)
    if desconocidas:
        raise ValueError(f"Etiquetas de frecuencia no reconocidas: {sorted(desconocidas)}")
    return serie.map(MESES_ENTRE_OBSERVACIONES)


def limpiar_rfqs(rfqs: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones de formato y columnas derivadas. No filtra filas."""
    df = rfqs.copy()

    for columna in COLUMNAS_FECHA:
        df[columna] = pd.to_datetime(df[columna])

    df["meses_entre_obs"] = normalizar_frecuencia(df["observation_frequency"])

    # Diferencia de meses de calendario, no de dias: los plazos se pactan en
    # meses redondos. Es un termino del contrato conocido al cotizar.
    df["vida_nominal_meses"] = (
        (df["end_date"].dt.year - df["start_date"].dt.year) * 12
        + (df["end_date"].dt.month - df["start_date"].dt.month)
    )

    df["n_subyacentes"] = df["underlyings"].str.count(r"\|") + 1

    return df


def seleccionar_entrenables(rfqs: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Filtra a RFQs ejecutadas y con duracion <= vida nominal."""
    df = rfqs[rfqs["executed"]].copy()
    coherentes = df["avg_duration_months"] <= df["vida_nominal_meses"]

    if verbose:
        print(f"RFQs totales:  {len(rfqs):>6}")
        print(f"  ejecutadas:  {len(df):>6}")
        print(f"  incoherentes descartadas: {(~coherentes).sum():>4}")
        print(f"  entrenables: {coherentes.sum():>6}")

    return df[coherentes].copy()


def preparar(ruta: Path = DATA, verbose: bool = True):
    rfqs, volatilidad, referencia = cargar(ruta)

    rfqs = limpiar_rfqs(rfqs)
    volatilidad["date"] = pd.to_datetime(volatilidad["date"])

    return seleccionar_entrenables(rfqs, verbose=verbose), volatilidad, referencia