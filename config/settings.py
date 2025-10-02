"""
Configuración del sistema multiagente con Ollama local.

Este módulo define la configuración centralizada usando Pydantic Settings
para validación y type safety.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración del sistema multiagente con Ollama local."""

    # Configuración de Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "gpt-oss:20b"
    ollama_embedding_model: str = "embeddinggemma"

    # Configuración de Google ADK
    app_name: str = "multiagent-system"

    # RAG Configuration
    documents_path: str = "data/documents"
    rag_enabled: bool = True
    chroma_persist_directory: str = "data/chroma_db"
    chroma_collection_name: str = "documents"

    # Web Search Configuration
    web_search_enabled: bool = False

    # Session Configuration
    session_timeout_seconds: int = 3600

    # Logging Configuration
    log_level: str = "INFO"

    # GPU Configuration
    require_gpu: bool = True
    cuda_visible_devices: str = "0"

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """
    Obtiene instancia de configuración cacheada.

    Returns:
        Settings: Instancia singleton de configuración
    """
    return Settings()
