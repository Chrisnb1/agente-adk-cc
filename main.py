"""Entry point principal del sistema multiagente."""

import asyncio
import logging
from pathlib import Path

from google.adk import Runner
from google.adk.sessions import InMemorySessionService

from src.agent.orchestrator import create_orchestrator_agent
from src.agent.sub_agents.rag_agent import create_rag_agent
from src.agent.sub_agents.web_agent import create_web_agent
from src.config.settings import get_settings
from src.rag.document_loader import load_documents
from src.rag.vector_store import VectorStoreManager
from src.utils.system_check import validate_system_requirements

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def initialize_system() -> tuple[Runner, VectorStoreManager]:
    """
    Inicializa el sistema multiagente.

    Returns:
        Tupla con runner del orquestador y vector store manager
    """
    logger.info("=== Iniciando sistema multiagente ===")

    # 1. Cargar configuración
    settings = get_settings()
    logger.info(f"Configuración cargada: {settings.app_name}")

    # 2. Validar prerequisitos
    logger.info("Validando prerequisitos del sistema...")
    await validate_system_requirements(settings)

    # 3. Inicializar vector store
    logger.info("Inicializando vector store...")
    vector_store_manager = VectorStoreManager(settings)

    # Verificar si ya existe vector store
    chroma_dir = Path(settings.chroma_persist_directory)
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        logger.info("Cargando vector store existente...")
        vector_store_manager.load_existing()
    else:
        logger.info("Creando nuevo vector store...")
        documents = load_documents(settings)
        vector_store_manager.initialize(documents)

    # 4. Crear agentes
    logger.info("Creando agentes...")
    rag_agent = await create_rag_agent(vector_store_manager, settings)
    web_agent = await create_web_agent(settings)
    orchestrator = await create_orchestrator_agent(rag_agent, web_agent, settings)

    # 5. Crear runner
    session_service = InMemorySessionService()
    runner = Runner(
        app_name=settings.app_name,
        agent=orchestrator,
        session_service=session_service,
    )

    logger.info("=== Sistema inicializado correctamente ===\n")
    return runner, vector_store_manager


async def run_interactive_mode(runner: Runner) -> None:
    """
    Ejecuta el sistema en modo interactivo.

    Args:
        runner: Runner del agente orquestador
    """
    print("\n" + "=" * 60)
    print("Sistema Multiagente con Google ADK + Ollama")
    print("=" * 60)
    print("Comandos disponibles:")
    print("  - Escribe tu pregunta y presiona Enter")
    print("  - 'exit' o 'quit' para salir")
    print("=" * 60 + "\n")

    while True:
        try:
            # Leer input del usuario
            user_input = input("Usuario: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("\n¡Hasta luego!")
                break

            if not user_input:
                continue

            # Ejecutar query
            print("\nOrquestador: Procesando...\n")
            response = await runner.run(user_input)
            print(f"Respuesta: {response.text}\n")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            logger.error(f"Error ejecutando query: {e}")
            print(f"\nError: {e}\n")


async def main() -> None:
    """Función principal."""
    try:
        # Inicializar sistema
        runner, vector_store_manager = await initialize_system()

        # Ejecutar en modo interactivo
        await run_interactive_mode(runner)

    except Exception as e:
        logger.error(f"Error fatal: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
