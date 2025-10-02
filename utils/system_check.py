"""
Validación de prerequisitos del sistema.

Este módulo verifica que todos los requisitos del sistema estén cumplidos
antes de iniciar el sistema multiagente.
"""

import subprocess

import requests

from src.config.settings import Settings
from src.exceptions.exceptions import (
    GPUNotAvailableException,
    ModelNotAvailableException,
    OllamaNotRunningException,
)


def check_gpu_available() -> bool:
    """
    Verifica si GPU NVIDIA está disponible.

    Returns:
        True si GPU disponible, False si no
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False, timeout=10
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        return False


def check_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """
    Verifica si Ollama está corriendo.

    Args:
        base_url: URL base de Ollama API

    Returns:
        True si Ollama responde, False si no
    """
    try:
        response = requests.get(base_url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def check_ollama_models(models: list[str], base_url: str) -> dict[str, bool]:
    """
    Verifica si modelos de Ollama están disponibles.

    Args:
        models: Lista de nombres de modelos
        base_url: URL base de Ollama API

    Returns:
        Dict con modelo -> disponible (True/False)
    """
    availability: dict[str, bool] = {}
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            available_models = [m["name"] for m in response.json().get("models", [])]
            for model in models:
                # Check both with and without :latest suffix
                model_base = model.split(":")[0]
                availability[model] = any(
                    m.startswith(model_base) or m == model for m in available_models
                )
        else:
            availability = dict.fromkeys(models, False)
    except requests.exceptions.RequestException:
        availability = dict.fromkeys(models, False)

    return availability


async def validate_system_requirements(settings: Settings) -> None:
    """
    Valida que todos los requisitos del sistema estén cumplidos.

    Args:
        settings: Configuración del sistema

    Raises:
        GPUNotAvailableException: Si GPU requerida pero no disponible
        OllamaNotRunningException: Si Ollama no está corriendo
        ModelNotAvailableException: Si modelos requeridos no están descargados
    """
    # CRÍTICO: Validar GPU si es requerida
    if settings.require_gpu:
        if not check_gpu_available():
            raise GPUNotAvailableException(
                "GPU NVIDIA no disponible. Verificar: nvidia-smi. Driver version >= 531 requerido."
            )
        print("✓ GPU NVIDIA disponible")

    # CRÍTICO: Validar Ollama corriendo
    if not check_ollama_running(settings.ollama_base_url):
        raise OllamaNotRunningException(
            f"Ollama no está corriendo en {settings.ollama_base_url}. Iniciar con: ollama serve"
        )
    print("✓ Ollama corriendo")

    # CRÍTICO: Validar modelos descargados
    required_models = [settings.ollama_llm_model, settings.ollama_embedding_model]
    model_status = check_ollama_models(required_models, settings.ollama_base_url)

    missing_models = [m for m, available in model_status.items() if not available]
    if missing_models:
        instructions = "\n".join([f"  ollama pull {model}" for model in missing_models])
        raise ModelNotAvailableException(
            f"Modelos faltantes: {', '.join(missing_models)}\nDescargar con:\n{instructions}"
        )

    print(f"✓ Modelos disponibles: {', '.join(required_models)}")
