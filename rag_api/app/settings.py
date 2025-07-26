from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------- Configuração do pydantic‑settings ----------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ---------- Caminhos base ----------
    # Diretório raiz do projeto = .../rag_api
    # (__file__ → .../rag_api/app/settings.py)
    app_root: Path = Path(__file__).resolve().parent.parent  # sobe 2 níveis

    # Mesmos caminhos que o CLI usa
    index_path: Path       = app_root / "source" / "models" / "odas_index_1606.faiss"
    meta_odas_path: Path   = app_root / "source" / "models" / "metadados_odas_1606.pkl"
    devolutivas_path: Path = app_root / "source" / "data"   / "devolutivas.csv"
    rubricas_path: Path    = app_root / "source" / "data"   / "rubricas.csv"

    # Token de autenticação
    api_token: str = "sk-yP7gL0bXfT29vqHkJREcA1NzWuK4qDms"


# Instância singleton usada pelo resto da aplicação
settings = Settings()
