from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI
from pydantic import BaseModel

# Configure import path for modules in streamlit_app/src
ROOT = Path(__file__).resolve().parent.parent
STREAMLIT_PATH = ROOT / "streamlit_app"
if str(STREAMLIT_PATH) not in sys.path:
    sys.path.append(str(STREAMLIT_PATH))

from src.utils import (
    carregar_modelo_st,
    carregar_index,
    carregar_metadados,
    carregar_devolutivas,
    carregar_rubricas,
    gerar_texto_devolutiva_markdown,
    gerar_texto_devolutiva_rico,
)
from src.recommendation import get_recommendations

app = FastAPI(title="RAG Gestores API")


class DevolutivaRequest(BaseModel):
    pontuacao: int
    dimensao: str
    subdimensao: str


class DevolutivaResponse(BaseModel):
    markdown: str | None


class RecommendationRequest(BaseModel):
    pontuacao: int
    dimensao: str
    subdimensao: str
    modelo: str = "Modelo Avançado (v2, Re-ranking)"


class RecommendationResponse(BaseModel):
    artigos: List[Dict]
    videos: List[Dict]
    audios: List[Dict]
    visuais: List[Dict]
    interativos: List[Dict]


@app.post("/devolutiva", response_model=DevolutivaResponse)
def generate_devolutiva(req: DevolutivaRequest) -> DevolutivaResponse:
    df_dev = carregar_devolutivas()
    df_rub = carregar_rubricas()
    markdown = gerar_texto_devolutiva_markdown(
        df_dev, df_rub, req.pontuacao, req.dimensao, req.subdimensao
    )
    return DevolutivaResponse(markdown=markdown)


@app.post("/recommendation", response_model=RecommendationResponse)
def recommendation(req: RecommendationRequest) -> RecommendationResponse:
    if req.modelo == "Modelo Antigo (Legacy)":
        index = carregar_index("models/odas_index_stellav5.faiss")
        df_odas = carregar_metadados("models/metadados_odas_stellav5.pkl")
    elif req.modelo == "Modelo Intermediário (Busca Simples)":
        index = carregar_index("models/odas_index_1606.faiss")
        df_odas = carregar_metadados("models/metadados_odas_1606.pkl")
    else:
        index = carregar_index("models/odas_index_1606_v2.faiss")
        df_odas = carregar_metadados("models/metadados_odas_1606_v2.pkl")

    modelo_st = carregar_modelo_st()
    df_dev = carregar_devolutivas()
    df_rub = carregar_rubricas()

    texto_rico = gerar_texto_devolutiva_rico(
        df_dev, df_rub, req.pontuacao, req.dimensao, req.subdimensao, req.modelo
    )
    if not texto_rico:
        return RecommendationResponse(
            artigos=[], videos=[], audios=[], visuais=[], interativos=[]
        )

    artigos, videos, audios, visuais, interativos = get_recommendations(
        modelo_st,
        index,
        df_odas,
        df_rub,
        req.pontuacao,
        req.dimensao,
        req.subdimensao,
        texto_rico,
        req.modelo,
    )

    return RecommendationResponse(
        artigos=artigos.to_dict(orient="records"),
        videos=videos.to_dict(orient="records"),
        audios=audios.to_dict(orient="records"),
        visuais=visuais.to_dict(orient="records"),
        interativos=interativos.to_dict(orient="records"),
    )

