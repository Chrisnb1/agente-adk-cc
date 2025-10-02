"""Configuración de embeddings con Ollama."""

from langchain_ollama import OllamaEmbeddings

from src.config.settings import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> OllamaEmbeddings:
    """
    Obtiene instancia de embeddings de Ollama.

    Args:
        settings: Configuración del sistema (opcional)

    Returns:
        OllamaEmbeddings: Instancia configurada de embeddings
    """
    if settings is None:
        settings = get_settings()

    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
