from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    pontuacao:   int   = Field(..., ge=0, description="Pontuação total obtida pelo usuário")
    dimensao:    str   = Field(..., description="Dimensão avaliada")
    subdimensao: str   = Field(..., description="Subdimensão avaliada")


class ODAItem(BaseModel):
    titulo:     str
    fonte:      Optional[str] = None
    distancia:  float
    url:        Optional[str] = None   # caso exista na base


class RecommendationsResponse(BaseModel):
    pontuacao:     int
    dimensao:      str
    subdimensao:   str
    rubrica_numero:int | None = None
    rubrica_nome:  str | None = None
    rubrica_tipo:  str | None = None
    devolutiva:    str | None = None
    recomendacoes: Dict[str, List[ODAItem]]
