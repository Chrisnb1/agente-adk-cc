"""
Excepciones personalizadas para el sistema multiagente.

Este módulo define todas las excepciones específicas del dominio para
manejo de errores robusto y descriptivo.
"""


class AgentException(Exception):
    """Excepción base para errores relacionados con agentes."""

    pass


class VectorStoreException(AgentException):
    """Se lanza cuando hay problemas con ChromaDB o el vector store."""

    pass


class DocumentLoaderException(AgentException):
    """Se lanza cuando hay problemas cargando documentos."""

    pass


class ConfigurationException(AgentException):
    """Se lanza cuando hay errores de configuración."""

    pass


class GPUNotAvailableException(AgentException):
    """
    Se lanza cuando GPU NVIDIA no está disponible pero es requerida.

    Attributes:
        message: Descripción del error con sugerencias de solución
    """

    def __init__(self, message: str) -> None:
        """
        Inicializa la excepción con un mensaje descriptivo.

        Args:
            message: Descripción del error
        """
        self.message = message
        super().__init__(self.message)


class OllamaNotRunningException(AgentException):
    """
    Se lanza cuando el servicio Ollama no está corriendo o no responde.

    Attributes:
        message: Descripción del error con sugerencias de solución
    """

    def __init__(self, message: str) -> None:
        """
        Inicializa la excepción con un mensaje descriptivo.

        Args:
            message: Descripción del error
        """
        self.message = message
        super().__init__(self.message)


class ModelNotAvailableException(AgentException):
    """
    Se lanza cuando un modelo requerido no está descargado en Ollama.

    Attributes:
        message: Descripción del error con instrucciones para descargar
    """

    def __init__(self, message: str) -> None:
        """
        Inicializa la excepción con un mensaje descriptivo.

        Args:
            message: Descripción del error
        """
        self.message = message
        super().__init__(self.message)
