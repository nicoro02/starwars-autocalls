"""API de inferencia.

Separa las dos fases del ciclo de vida del modelo:

    entrenamiento  lento, se ejecuta de forma puntual, necesita el historico
                   completo  ->  produce un artefacto en disco
    inferencia     rapida, se ejecuta en cada cotizacion, solo necesita ese
                   artefacto  ->  lo consume

La API carga el modelo UNA vez al arrancar, no en cada peticion. Recibe los
terminos de una solicitud tal y como llegan de front-office y construye
internamente las mismas features que se usaron al entrenar, reutilizando los
modulos del pipeline: si la transformacion se duplicara aqui, entrenamiento e
inferencia podrian divergir sin que nadie lo notase.

Levantar en local:
    uvicorn src.api:app --reload
Documentacion interactiva:
    http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from datetime import date

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import DATA, MESES_ENTRE_OBSERVACIONES, cargar, limpiar_rfqs
from src.features import construir

RUTA_MODELO = DATA.parent / "models" / "modelo.joblib"

# Estado cargado al arrancar el servicio
estado: dict = {}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carga el artefacto y los datos de mercado al arrancar."""
    artefacto = joblib.load(RUTA_MODELO)
    _, volatilidad, referencia = cargar()
    volatilidad["date"] = pd.to_datetime(volatilidad["date"])

    estado["artefacto"] = artefacto
    estado["volatilidad"] = volatilidad
    estado["referencia"] = referencia
    yield
    estado.clear()


app = FastAPI(
    title="Estimacion de duracion de autocallables",
    description="Predice la duracion media de un producto estructurado a partir "
                "de los terminos de la solicitud de cotizacion.",
    version="0.1.0",
    lifespan=ciclo_de_vida,
)


class Solicitud(BaseModel):
    """Terminos de una RFQ, tal y como los recibe la mesa al cotizar."""

    product_type: str = Field(..., examples=["Kessel Run Snowball"])
    underlyings: str = Field(..., description="Tickers separados por '|'",
                             examples=["CLNE|DRC"])
    basket_type: str = Field(..., examples=["worst_of"])
    autocall_barrier_pct: float = Field(..., gt=0, examples=[1.0])
    no_call_period_months: int = Field(..., ge=0, examples=[3])
    observation_frequency: str = Field(..., examples=["Monthly"])
    quoted_implied_vol: float = Field(..., gt=0, examples=[0.2455])
    requested_date: date = Field(..., examples=["2023-09-08"])
    start_date: date = Field(..., examples=["2023-09-08"])
    end_date: date = Field(..., description="Vencimiento nominal pactado",
                           examples=["2028-09-08"])


class Prediccion(BaseModel):
    duracion_media_meses: float
    vida_nominal_meses: int
    fraccion_vida_consumida: float


@app.get("/health")
def health():
    """Estado del servicio. Permite comprobar que el modelo esta cargado."""
    artefacto = estado.get("artefacto")
    return {
        "status": "ok" if artefacto else "sin modelo",
        "modelo_cargado": artefacto is not None,
        "n_features": len(artefacto["features"]) if artefacto else 0,
        "mae_test_meses": round(artefacto["metricas_test"]["MAE"], 2) if artefacto else None,
    }


@app.post("/predict", response_model=Prediccion)
def predict(solicitud: Solicitud):
    """Estima la duracion media del producto solicitado."""
    artefacto = estado.get("artefacto")
    if artefacto is None:
        raise HTTPException(503, "El modelo no esta cargado")

    if solicitud.observation_frequency not in MESES_ENTRE_OBSERVACIONES:
        raise HTTPException(
            422,
            f"Frecuencia no reconocida: '{solicitud.observation_frequency}'. "
            f"Valores admitidos: {sorted(MESES_ENTRE_OBSERVACIONES)}",
        )

    datos = solicitud.model_dump()
    # Pydantic devuelve objetos date, que pandas convierte a una resolucion
    # temporal distinta a la de la tabla de volatilidad y romperia el
    # merge_asof. Se pasan como texto para que el parseo sea el mismo que en
    # el entrenamiento.
    for campo in ("requested_date", "start_date", "end_date"):
        datos[campo] = datos[campo].isoformat()

    entrada = pd.DataFrame([datos])

    try:
        # Mismas transformaciones que en entrenamiento, importadas del pipeline
        entrada = limpiar_rfqs(entrada)
        entrada = construir(entrada.assign(rfq_id="consulta"),
                            estado["volatilidad"], estado["referencia"])
    except ValueError as e:
        raise HTTPException(422, str(e))

    pred = artefacto["modelo"].predict(entrada[artefacto["features"]])[0]

    vida = int(entrada["vida_nominal_meses"].iloc[0])
    if artefacto["usar_ratio"]:
        fraccion, meses = pred, pred * vida
    else:
        meses, fraccion = pred, pred / vida

    # La duracion no puede superar la vida nominal del producto: la misma
    # restriccion fisica que se aplico al limpiar los datos de entrenamiento.
    meses = min(meses, vida)

    return Prediccion(
        duracion_media_meses=round(float(meses), 2),
        vida_nominal_meses=vida,
        fraccion_vida_consumida=round(float(min(fraccion, 1.0)), 4),
    )