from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .settings import settings
from .models import RecommendationRequest, RecommendationsResponse
from . import rag_service


app = FastAPI(
    title="RAG – API de Recomendações",
    version="1.0.0",
    description="Gera devolutivas e recomendações pedagógicas a partir de inputs.",
)

security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency que valida o token Bearer."""
    if credentials.credentials != settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token inválido.",
        )


@app.on_event("startup")
def _startup():
    rag_service.init()  # garante preload


@app.post(
    "/recommend",
    response_model=RecommendationsResponse,
    dependencies=[Depends(verify_token)],
    summary="Gera devolutiva + recomendações",
)
def recommend(req: RecommendationRequest):
    """Endpoint principal."""
    payload = rag_service.recommend(req.pontuacao, req.dimensao, req.subdimensao)
    return payload
