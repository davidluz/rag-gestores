from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------- Configuração do pydantic-settings ----------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ---------- Caminhos base ----------
    # Diretório raiz = .../rag-gestores
    app_root: Path = Path(__file__).resolve().parent.parent.parent

    # Mesmos caminhos que o CLI usa
    index_path: Path       = app_root / "streamlit_app" / "models" / "odas_index_1606.faiss"
    meta_odas_path: Path   = app_root / "streamlit_app" / "models" / "metadados_odas_1606.pkl"
    devolutivas_path: Path = app_root / "streamlit_app" / "data"   / "devolutivas.csv"
    rubricas_path: Path    = app_root / "streamlit_app" / "data"   / "rubricas.csv"


    # Token de autenticação
    api_token: str = "CHANGE_ME"


# Instância singleton usada pelo resto da aplicação
settings = Settings()
