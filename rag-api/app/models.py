from pydantic import BaseModel, Field
from typing import List

class InputPayload(BaseModel):
    pontuacao: float = Field(..., example=7.5)
    subdimensao: str = Field(..., example="Motivação")
    preferencias_do_usuario: List[str] = Field(..., example=["vídeo", "podcast"])
    dimensao: str = Field(..., example="Empatia")

class OutputPayload(BaseModel):
    devolutiva: str
    materiais: List[str]
