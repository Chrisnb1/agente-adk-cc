"""Cargador de documentos para RAG."""

import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from src.config.settings import Settings, get_settings
from src.exceptions.exceptions import DocumentLoaderException

logger = logging.getLogger(__name__)


def load_documents(settings: Settings | None = None) -> list[Document]:
    """
    Carga documentos desde el directorio configurado.

    Args:
        settings: Configuración del sistema (opcional)

    Returns:
        Lista de documentos cargados

    Raises:
        DocumentLoaderException: Si hay problemas cargando documentos
    """
    if settings is None:
        settings = get_settings()

    documents_path = Path(settings.documents_path)

    if not documents_path.exists():
        raise DocumentLoaderException(f"Documents path does not exist: {documents_path}")

    if not documents_path.is_dir():
        raise DocumentLoaderException(f"Documents path is not a directory: {documents_path}")

    documents: list[Document] = []
    text_files = list(documents_path.glob("**/*.txt"))

    if not text_files:
        logger.warning(f"No .txt files found in {documents_path}")
        return documents

    for file_path in text_files:
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            file_docs = loader.load()
            documents.extend(file_docs)
            logger.info(f"Loaded {len(file_docs)} documents from {file_path.name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise DocumentLoaderException(f"Failed to load document {file_path}: {e}") from e

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents
