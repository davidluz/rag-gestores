from fastapi import APIRouter
from pydantic import BaseModel
from src.utils.feedback import selecionar_devolutiva  # Importa a função

router = APIRouter()

class RagInput(BaseModel):
    usuario_id: int
    score: float  # Recebe o score direto

@router.post("/processar")
def processar_rag(input_data: RagInput):
    feedback = selecionar_devolutiva(input_data.score)
    return {"usuario_id": input_data.usuario_id, "feedback": feedback}
