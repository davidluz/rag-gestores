from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import InputPayload, OutputPayload
from .rag_service import RAGService

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

rag_service = RAGService()

@app.post("/gerar", response_model=OutputPayload)
async def gerar(payload: InputPayload):
    try:
        texto, materiais = rag_service.infer(payload.dict())
        return OutputPayload(devolutiva=texto, materiais=materiais)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
