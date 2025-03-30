from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Modelo de entrada para a requisição de RAG
class RagInput(BaseModel):
    usuario_id: int
    respostas: dict  # Ex.: {"competencia1": 2.5, "competencia2": 3.8, ...}

@router.post("/processar")
def processar_rag(input_data: RagInput):
    # Aqui, você vai integrar os módulos de pré, recuperação, etc.
    # Por enquanto, retorna uma resposta mockada
    feedback = "Feedback integrado para o gestor."
    return {"usuario_id": input_data.usuario_id, "feedback": feedback}
