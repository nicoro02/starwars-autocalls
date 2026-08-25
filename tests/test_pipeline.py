"""Tests del pipeline.

Cubren sobre todo fallos silenciosos: no producen error, solo resultados
incorrectos que parecerian validos.

Ejecutar con:
    pytest -v
"""

import pandas as pd
import pytest

from src.data import MESES_ENTRE_OBSERVACIONES, limpiar_rfqs, normalizar_frecuencia, preparar
from src.features import FEATURES, construir, unir_volatilidad_pit


@pytest.fixture(scope="module")
def datos():
    return preparar(verbose=False)


# ---------------------------------------------------------------------------
# 1. Fuga temporal
# ---------------------------------------------------------------------------

def test_volatilidad_nunca_posterior_a_la_rfq(datos):
    """El cruce con mercado solo puede usar datos anteriores a la solicitud.

    daily_volatility llega hasta 2026 y la ultima RFQ es de 2024: un merge sin
    restriccion temporal usaria informacion del futuro. El fallo no daria
    error, solo metricas artificialmente buenas.
    """
    rfqs, volatilidad, _ = datos

    cesta = (
        rfqs[["rfq_id", "requested_date", "underlyings"]]
        .assign(underlying=lambda d: d["underlyings"].str.split("|"))
        .explode("underlying")
        .drop(columns="underlyings")
    )
    unido = unir_volatilidad_pit(cesta, volatilidad)

    assert (unido["date"] <= unido["requested_date"]).all()


def test_detecta_fuga_si_se_invierte_la_direccion(datos):
    """El control anterior debe fallar si el cruce mirase hacia adelante.

    Sin esta comprobacion no sabriamos si el test anterior pasa porque el
    cruce es correcto o porque nunca podria fallar.
    """
    rfqs, volatilidad, _ = datos

    cesta = (
        rfqs[["rfq_id", "requested_date", "underlyings"]]
        .head(500)
        .assign(underlying=lambda d: d["underlyings"].str.split("|"))
        .explode("underlying")
        .drop(columns="underlyings")
        .sort_values("requested_date")
    )

    futuro = pd.merge_asof(
        cesta,
        volatilidad.sort_values("date"),
        left_on="requested_date",
        right_on="date",
        by="underlying",
        # Con allow_exact_matches=False se fuerza a tomar estrictamente la
        # siguiente cotizacion: informacion posterior a la solicitud.
        allow_exact_matches=False,
        direction="forward",
    )
    assert not (futuro["date"] <= futuro["requested_date"]).all()


# ---------------------------------------------------------------------------
# 2. Etiquetas de frecuencia
# ---------------------------------------------------------------------------

def test_todas_las_etiquetas_se_traducen(datos):
    """Las 18 etiquetas del historico deben mapear a 6 frecuencias reales."""
    rfqs, _, _ = datos
    assert rfqs["meses_entre_obs"].notna().all()
    assert rfqs["meses_entre_obs"].nunique() == 6


def test_sinonimos_dan_el_mismo_valor():
    """'1M', 'Monthly', 'mensual', '1 month' y 'M' son la misma frecuencia."""
    mensuales = ["1M", "M", "Monthly", "mensual", "1 month"]
    valores = {MESES_ENTRE_OBSERVACIONES[e] for e in mensuales}
    assert valores == {1.0}


def test_etiqueta_desconocida_falla_en_lugar_de_generar_nulo():
    """Ante una frecuencia nueva el proceso debe detenerse, no predecir mal.

    Un .map() silencioso devolveria NaN y el modelo seguiria funcionando con
    una feature vacia.
    """
    serie = pd.Series(["1M", "cada dos martes"])
    with pytest.raises(ValueError, match="no reconocidas"):
        normalizar_frecuencia(serie)


# ---------------------------------------------------------------------------
# 3. Coherencia entre entrenamiento e inferencia
# ---------------------------------------------------------------------------

def test_una_rfq_suelta_produce_las_mismas_features(datos):
    """Entrenamiento e inferencia deben construir features identicas.

    La API procesa las solicitudes de una en una y el entrenamiento en lote.
    Si las dos rutas divergieran, el modelo predeciria sobre variables
    distintas de las que aprendio, sin dar ningun error.
    """
    rfqs, volatilidad, referencia = datos

    lote = construir(rfqs.head(50), volatilidad, referencia)
    suelta = construir(rfqs.head(50).iloc[[7]], volatilidad, referencia)

    esperado = lote[FEATURES].iloc[[7]].reset_index(drop=True)
    obtenido = suelta[FEATURES].reset_index(drop=True)

    pd.testing.assert_frame_equal(esperado, obtenido)


def test_no_hay_nulos_en_las_features(datos):
    """Ninguna feature puede llegar vacia al modelo."""
    rfqs, volatilidad, referencia = datos
    df = construir(rfqs, volatilidad, referencia)
    assert df[FEATURES].isna().sum().sum() == 0


def test_subyacente_desconocido_falla(datos):
    """Un ticker sin datos de mercado debe detener el proceso."""
    rfqs, volatilidad, referencia = datos
    fila = rfqs.head(1).copy()
    fila["underlyings"] = "XXXX"

    with pytest.raises(ValueError):
        construir(fila, volatilidad, referencia)


# ---------------------------------------------------------------------------
# Coherencia de negocio
# ---------------------------------------------------------------------------

def test_la_duracion_no_supera_la_vida_nominal(datos):
    """Restriccion fisica del producto sobre la muestra de entrenamiento."""
    rfqs, _, _ = datos
    assert (rfqs["avg_duration_months"] <= rfqs["vida_nominal_meses"]).all()


def test_n_oportunidades_es_positivo(datos):
    """El periodo de no-call nunca puede consumir toda la vida del producto."""
    rfqs, volatilidad, referencia = datos
    df = construir(rfqs, volatilidad, referencia)
    assert (df["n_oportunidades"] > 0).all()