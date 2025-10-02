"""RAG Agent con ChromaDB y Ollama."""

import logging

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from src.config.settings import Settings, get_settings
from src.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


async def create_rag_agent(
    vector_store_manager: VectorStoreManager, settings: Settings | None = None
) -> Agent:
    """
    Crea un agente RAG con ChromaDB.

    Args:
        vector_store_manager: Gestor de vector store
        settings: Configuración del sistema (opcional)

    Returns:
        Agente RAG configurado
    """
    if settings is None:
        settings = get_settings()

    model = LiteLlm(
        model=f"ollama_chat/{settings.ollama_llm_model}",
        api_base=settings.ollama_base_url,
    )

    system_prompt = """Eres un asistente especializado en responder preguntas usando información de documentos.

Tu responsabilidad:
- Buscar información relevante en la base de conocimiento
- Responder preguntas basándote SOLO en la información encontrada
- Citar las fuentes cuando sea apropiado
- Admitir cuando no tienes información suficiente

Cuando respondas:
1. Busca en la base de conocimiento información relevante
2. Analiza los resultados obtenidos
3. Responde de forma clara y concisa
4. Si no encuentras información, dilo claramente"""

    agent = Agent(
        model=model,
        name="rag_agent",
        description="Agente especializado en búsqueda y recuperación de información",
        instruction=system_prompt,
    )

    logger.info("RAG Agent created successfully")
    return agent


async def query_rag_agent(
    agent: Agent, vector_store_manager: VectorStoreManager, query: str, k: int = 4
) -> str:
    """
    Consulta al agente RAG con contexto de documentos.

    Args:
        agent: Instancia del agente RAG
        vector_store_manager: Gestor de vector store
        query: Pregunta del usuario
        k: Número de documentos a recuperar

    Returns:
        Respuesta del agente
    """
    # Buscar documentos relevantes
    relevant_docs = vector_store_manager.similarity_search(query, k=k)

    # Construir contexto
    context = "\n\n".join(
        [f"Documento {i + 1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)]
    )

    # Crear prompt con contexto
    prompt = f"""Contexto de la base de conocimiento:
{context}

Pregunta del usuario: {query}

Por favor responde basándote en el contexto proporcionado."""

    # Ejecutar agente
    response = await agent.run(prompt)
    return response
