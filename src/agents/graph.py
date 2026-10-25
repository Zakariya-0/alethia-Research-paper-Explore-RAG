"""
Module d'Assemblage du Graphe d'États Multi-Agents LangGraph.

Rôle dans le projet :
----------------------
Orchestre l'ensemble du pipeline cognitif multi-agents sous forme d'un StateGraph.
Définit la topologie du graphe, les transitions séquentielles, et l'arête conditionnelle
de rétroaction active (Active Feedback Loop).

Topologie du Graphe :
[START]
   │
   ▼
[Agent Analyseur] ◄─────────────────────────────────┐ (Boucle de rétroaction active)
   │                                                │
   ▼                                                │
[Agent Récupérateur]                                │
   │                                                │
   ▼                                                │
[Agent Évaluateur]                                  │
   │                                                │
   ├─► (needs_feedback == True & retries < max) ────┘
   │
   ▼ (Preuves suffisantes OU max itérations atteint)
[Agent Générateur]  (réponse rédigée par le LLM ; ignoré si LLM inactif)
   │
   ▼
[Agent Vérificateur]  (audit des citations, dont celles du LLM)
   │
   ▼
[Agent Synthèse]
   │
   ▼
 [END]
"""

import sys
import logging
from typing import Dict, Any, Literal

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langgraph.graph import StateGraph, START, END

from src.agents.state import VerificationState, create_initial_state
from src.agents.nodes import (
    analyzer_node,
    retriever_node,
    evaluator_node,
    generator_node,
    verifier_node,
    synthesis_node,
)

logger = logging.getLogger(__name__)


def route_evaluator_decision(
    state: VerificationState,
) -> Literal["analyseur", "generateur"]:
    """
    Fonction de routage conditionnel (Conditional Edge) :
    - Si l'Agent Évaluateur a positionné 'needs_feedback=True', on reboucle vers l'Agent Analyseur.
    - Sinon, on continue le flux nominal vers l'Agent Générateur.
    """
    if state.get("needs_feedback", False):
        logger.info(
            f"🔄 Rétroaction active déclenchée : {state.get('feedback_reason')}"
        )
        return "analyseur"
    return "generateur"


def build_verification_graph() -> Any:
    """
    Construit et compile le StateGraph LangGraph complet.

    Returns:
        Graph compilé prêt à être exécuté avec .invoke().
    """
    # 1. Instanciation du graphe d'états
    workflow = StateGraph(VerificationState)

    # 2. Enregistrement des 6 nœuds agents
    workflow.add_node("analyseur", analyzer_node)
    workflow.add_node("recuperateur", retriever_node)
    workflow.add_node("evaluateur", evaluator_node)
    workflow.add_node("generateur", generator_node)
    workflow.add_node("verificateur", verifier_node)
    workflow.add_node("synthese", synthesis_node)

    # 3. Définition des transitions de flux
    workflow.add_edge(START, "analyseur")
    workflow.add_edge("analyseur", "recuperateur")
    workflow.add_edge("recuperateur", "evaluateur")

    # 4. Arête conditionnelle pour la boucle de rétroaction active
    workflow.add_conditional_edges(
        "evaluateur",
        route_evaluator_decision,
        {"analyseur": "analyseur", "generateur": "generateur"},
    )

    workflow.add_edge("generateur", "verificateur")
    workflow.add_edge("verificateur", "synthese")
    workflow.add_edge("synthese", END)

    # 5. Compilation du graphe
    compiled_graph = workflow.compile()
    logger.info("Graphe multi-agents LangGraph assemblé et compilé avec succès.")
    return compiled_graph


# Instance compilée partagée
verification_pipeline = build_verification_graph()


def verify_claim_agentic(
    claim: str, max_feedback_iterations: int = 2
) -> Dict[str, Any]:
    """
    Point d'entrée de haut niveau pour exécuter la vérification multi-agents.

    Args:
        claim: L'affirmation scientifique à vérifier.
        max_feedback_iterations: Plafond de boucles de réessaie autorisées.

    Returns:
        État final complet après exécution de tous les agents.
    """
    initial_state = create_initial_state(
        claim=claim, max_feedback_iterations=max_feedback_iterations
    )

    final_state = verification_pipeline.invoke(initial_state)
    return final_state
