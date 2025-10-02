"""Web Agent con conocimiento general."""

import logging

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


async def create_web_agent(settings: Settings | None = None) -> Agent:
    """
    Crea un agente Web con conocimiento general.

    Args:
        settings: Configuración del sistema (opcional)

    Returns:
        Agente Web configurado
    """
    if settings is None:
        settings = get_settings()

    model = LiteLlm(
        model=f"ollama_chat/{settings.ollama_llm_model}",
        api_base=settings.ollama_base_url,
    )

    system_prompt = """Eres un asistente especializado en proporcionar información general y conocimiento amplio.

Tu responsabilidad:
- Responder preguntas de conocimiento general
- Proporcionar explicaciones claras y concisas
- Mantener respuestas precisas y verificables
- Reconocer cuando una pregunta está fuera de tu alcance

Cuando respondas:
1. Proporciona respuestas directas y útiles
2. Usa ejemplos cuando sea apropiado
3. Estructura la información de forma clara
4. Admite cuando no estés seguro de algo"""

    root_agent = Agent(
        model=model,
        name="web_agent",
        description="Agente especializado en conocimiento general y consultas amplias",
        instruction=system_prompt,
    )

    logger.info("Web Agent created successfully")
    return root_agent
