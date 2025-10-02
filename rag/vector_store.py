"""Gestión de vector store con ChromaDB."""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config.settings import Settings, get_settings
from src.exceptions.exceptions import VectorStoreException
from src.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Gestor de vector store con ChromaDB."""

    def __init__(
        self, settings: Settings | None = None, embeddings: Embeddings | None = None
    ) -> None:
        """
        Inicializa el gestor de vector store.

        Args:
            settings: Configuración del sistema (opcional)
            embeddings: Instancia de embeddings (opcional)
        """
        self.settings = settings or get_settings()
        self.embeddings = embeddings or get_embeddings(self.settings)
        self.vector_store: Chroma | None = None

    def initialize(self, documents: list[Document]) -> None:
        """
        Inicializa el vector store con documentos.

        Args:
            documents: Lista de documentos a indexar

        Raises:
            VectorStoreException: Si hay problemas inicializando
        """
        if not documents:
            raise VectorStoreException("No documents provided for initialization")

        try:
            persist_dir = Path(self.settings.chroma_persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.settings.chroma_collection_name,
                persist_directory=str(persist_dir),
            )
            logger.info(f"Vector store initialized with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise VectorStoreException(f"Failed to initialize vector store: {e}") from e

    def load_existing(self) -> None:
        """
        Carga un vector store existente.

        Raises:
            VectorStoreException: Si hay problemas cargando
        """
        try:
            persist_dir = Path(self.settings.chroma_persist_directory)

            if not persist_dir.exists():
                raise VectorStoreException(f"Persist directory does not exist: {persist_dir}")

            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                collection_name=self.settings.chroma_collection_name,
                persist_directory=str(persist_dir),
            )
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise VectorStoreException(f"Failed to load vector store: {e}") from e

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """
        Busca documentos similares a la query.

        Args:
            query: Query de búsqueda
            k: Número de resultados a retornar

        Returns:
            Lista de documentos similares

        Raises:
            VectorStoreException: Si hay problemas con la búsqueda
        """
        if self.vector_store is None:
            raise VectorStoreException("Vector store not initialized")

        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise VectorStoreException(f"Similarity search failed: {e}") from e

    def get_vector_store(self) -> Chroma:
        """
        Obtiene la instancia de vector store.

        Returns:
            Instancia de Chroma

        Raises:
            VectorStoreException: Si vector store no está inicializado
        """
        if self.vector_store is None:
            raise VectorStoreException("Vector store not initialized")
        return self.vector_store
