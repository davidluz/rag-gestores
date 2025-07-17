from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    model_path: str = "modelo_rag.pkl"
    class Config:
        env_file = ".env"
settings = Settings()
