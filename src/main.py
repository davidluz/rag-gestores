from fastapi import FastAPI
from src.routes.rag_routes import router  # Importação absoluta

app = FastAPI()

app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "API funcionando"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)


