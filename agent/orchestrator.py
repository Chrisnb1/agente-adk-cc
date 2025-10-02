"""Orchestrator Agent - Coordina los sub-agentes."""

import logging

from google.adk.agents import Agent
from google.adk.tools import AgentTool
from google.adk.models.lite_llm import LiteLlm

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


async def create_orchestrator_agent(
    rag_agent: Agent, web_agent: Agent, settings: Settings | None = None
) -> Agent:
    """
    Crea el agente orquestador que coordina los sub-agentes.

    Args:
        rag_agent: Agente RAG
        web_agent: Agente Web
        settings: Configuración del sistema (opcional)

    Returns:
        Agente orquestador configurado
    """
    if settings is None:
        settings = get_settings()

    model = LiteLlm(
        model=f"ollama_chat/{settings.ollama_llm_model}",
        api_base=settings.ollama_base_url,
    )

    system_prompt = """Eres un orquestador inteligente que coordina dos agentes especializados:

1. **rag_agent**: Especializado en buscar información en documentos específicos de la base de conocimiento
   - Usa este agente para preguntas sobre información contenida en documentos
   - Ideal para consultas técnicas, datos específicos o información documentada

2. **web_agent**: Especializado en conocimiento general y consultas amplias
   - Usa este agente para preguntas de conocimiento general
   - Ideal para explicaciones conceptuales, definiciones, o temas generales

Tu responsabilidad:
1. Analizar la pregunta del usuario
2. Decidir qué agente es más apropiado
3. Delegar la tarea al agente correcto
4. Presentar la respuesta tec usuario de forma clara

Directrices de selección:
- Si la pregunta parece requerir información de documentos específicos → usa rag_agent
- Si la pregunta es de conocimiento general o conceptual → usa web_agent
- Si tienes dudas, intenta primero con rag_agent, y si no hay resultados usa web_agent

Responde siempre de forma directa y útil."""

    # Crear tools para los agentes
    rag_tool = AgentTool(agent=rag_agent)
    web_tool = AgentTool(agent=web_agent)

    # static_instruction=types.Content

    orchestrator = Agent(
        model=model,
        name="orchestrator",
        description="Orquestador principal que coordina agentes especializados",
        instruction=system_prompt,
        tools=[rag_tool, web_tool],
    )

    logger.info("Orchestrator Agent created successfully")
    return orchestrator
