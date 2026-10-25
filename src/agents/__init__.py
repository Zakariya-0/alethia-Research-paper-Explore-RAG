"""
Package multi-agents LangGraph pour la vérification scientifique autonome.

Expose l'état global du graphe (VerificationState), les nœuds agents et le pipeline compilé.
"""

from .state import (
    VerificationState,
    AgentLogEntry,
    create_initial_state,
    format_agent_log,
)
from .nodes import (
    ModelRegistry,
    analyzer_node,
    retriever_node,
    evaluator_node,
    generator_node,
    verifier_node,
    synthesis_node,
)
from .graph import build_verification_graph, verification_pipeline, verify_claim_agentic

__all__ = [
    "VerificationState",
    "AgentLogEntry",
    "create_initial_state",
    "format_agent_log",
    "ModelRegistry",
    "analyzer_node",
    "retriever_node",
    "evaluator_node",
    "generator_node",
    "verifier_node",
    "synthesis_node",
    "build_verification_graph",
    "verification_pipeline",
    "verify_claim_agentic",
]
