"""ADK Web entrypoint - expone orchestrator como root_agent."""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import AgentTool

from src.config.settings import get_settings
from src.rag.vector_store import VectorStoreManager

settings = get_settings()
vector_store = VectorStoreManager(settings)
vector_store.load_existing()

model = LiteLlm(
    model=f"ollama_chat/{settings.ollama_llm_model}",
    api_base=settings.ollama_base_url,
)

# RAG Agent
rag_agent = Agent(
    model=model,
    name="rag_agent",
    description="Agente especializado en búsqueda de documentos",
    instruction="Eres un asistente especializado en buscar información en documentos.",
)

# Web Agent
web_agent = Agent(
    model=model,
    name="web_agent",
    description="Agente de conocimiento general",
    instruction="Eres un asistente de conocimiento general.",
)

# Orchestrator
root_agent = Agent(
    model=model,
    name="orchestrator",
    description="Orquestador principal que coordina agentes especializados",
    instruction="""Coordinas dos agentes: rag_agent para búsqueda en documentos y web_agent para conocimiento general. Decide cuál usar según la pregunta.""",
    tools=[AgentTool(agent=rag_agent), AgentTool(agent=web_agent)],
)
