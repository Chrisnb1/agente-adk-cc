# Sistema Multiagente con Google ADK + Ollama

Sistema de agentes especializados construido con Google ADK (Agent Development Kit) y Ollama para LLM local.

## Arquitectura

El sistema implementa un patrón de orquestación con tres agentes principales:

```
orchestrator
    ├── rag_agent (búsqueda en documentos)
    └── web_agent (conocimiento general)
```

### Agentes

- **orchestrator**: Coordina los sub-agentes y delega tareas según el tipo de consulta
- **rag_agent**: Especializado en búsqueda y recuperación de información en documentos usando ChromaDB
- **web_agent**: Proporciona respuestas de conocimiento general

## Componentes Principales

### [agent/](agent/)
Implementación de agentes ADK:
- [orchestrator.py](agent/orchestrator.py:14) - Agente coordinador principal
- [sub_agents/rag_agent.py](agent/sub_agents/rag_agent.py:14) - Agente RAG con vector store
- [sub_agents/web_agent.py](agent/sub_agents/web_agent.py:13) - Agente de conocimiento general

### [rag/](rag/)
Sistema RAG (Retrieval-Augmented Generation):
- [vector_store.py](rag/vector_store.py) - Gestión de ChromaDB
- [document_loader.py](rag/document_loader.py) - Carga de documentos
- [embeddings.py](rag/embeddings.py) - Generación de embeddings con Ollama

### [config/](config/)
Configuración del sistema:
- [settings.py](config/settings.py:13) - Configuración con Pydantic

### [main.py](main.py:24)
Entry point principal con modo interactivo

## Uso

### Ejecución del Sistema

```bash
# Modo interactivo
uv run python -m src.main
```

### Ejemplo de Consultas

```python
# Pregunta para RAG agent (documentos)
"¿Qué información hay sobre X en los documentos?"

# Pregunta para Web agent (conocimiento general)
"¿Qué es machine learning?"
```

## Configuración

Variables principales en [settings.py](config/settings.py):

```python
ollama_base_url: str = "http://localhost:11434"
ollama_llm_model: str = "gpt-oss:20b"
ollama_embedding_model: str = "embeddinggemma"
chroma_persist_directory: str = "src/data/chroma_db"
```

## Requisitos

- Python 3.12+
- Ollama ejecutándose localmente
- GPU NVIDIA (configurable)
- ChromaDB para vector store

## Arquitectura de Agentes ADK

El sistema usa Google ADK con los siguientes patrones:

1. **Agent**: Unidad básica con modelo LLM
2. **AgentTool**: Wrapper para delegar tareas entre agentes
3. **Runner**: Ejecutor de agentes con gestión de sesiones
4. **LiteLlm**: Integración con Ollama vía LiteLLM

Ver [examples/agent.py](../examples/agent.py:3) para ejemplo básico de configuración.
