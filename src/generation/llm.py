"""
Fabrique du LLM de la plateforme.

Le modèle est créé avec `langchain.chat_models.init_chat_model` : le fournisseur
(LLM_PROVIDER, ex. "groq") et le modèle (LLM_MODEL) viennent des variables
d'environnement, la clé API de la variable propre au fournisseur (GROQ_API_KEY).
Rien n'est codé en dur dans le pipeline : changer de fournisseur ne demande
aucune modification de code.
"""

import logging
import os
import threading
from typing import Any, Dict, Optional

from src.config import (
    LLM_API_KEY_VARS,
    LLM_ENABLED,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

_model: Optional[Any] = None
_model_lock = threading.Lock()


def llm_status() -> Dict[str, Any]:
    """
    État du LLM, exposé à l'API et à l'interface.

    Returns:
        {"enabled": bool, "provider": str, "model": str, "reason": str | None}
        `reason` explique pourquoi le LLM est inactif (désactivé ou clé absente).
    """
    reason: Optional[str] = None
    if not LLM_ENABLED:
        reason = "LLM désactivé (LLM_ENABLED=false)"
    else:
        key_var = LLM_API_KEY_VARS.get(LLM_PROVIDER)
        if key_var and not os.getenv(key_var):
            reason = f"Clé API absente : définissez {key_var} dans le fichier .env"

    return {
        "enabled": reason is None,
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "reason": reason,
    }


def get_chat_model() -> Any:
    """
    Retourne l'instance partagée du modèle de chat, créée au premier appel.

    Raises:
        RuntimeError: si le LLM est désactivé ou si la clé API est absente.
    """
    global _model

    status = llm_status()
    if not status["enabled"]:
        raise RuntimeError(status["reason"])

    if _model is None:
        with _model_lock:
            if _model is None:
                # Import local : le reste de l'application démarre sans langchain
                from langchain.chat_models import init_chat_model

                logger.info(f"Création du LLM {LLM_PROVIDER}:{LLM_MODEL}...")
                _model = init_chat_model(
                    LLM_MODEL,
                    model_provider=LLM_PROVIDER,
                    temperature=LLM_TEMPERATURE,
                    timeout=LLM_TIMEOUT_SECONDS,
                    max_retries=LLM_MAX_RETRIES,
                    max_tokens=LLM_MAX_TOKENS,
                )
    return _model
