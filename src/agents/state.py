"""
Module de Définition de l'État Global du Graphe Multi-Agents (LangGraph State).

Rôle dans le projet :
----------------------
Dans l'architecture multi-agents de LangGraph, l'état global (State) agit comme
le tableau de bord partagé entre tous les agents spécialisés :
1. Agent Analyseur (Query Analysis)
2. Agent Récupérateur (Retriever & Reranker)
3. Agent Évaluateur (NLI DeBERTa Logic)
4. Agent Vérificateur (Citation Anti-Hallucination)
5. Agent Synthèse (Confidence & Report Generator)

Chaque agent lit cet état, effectue sa tâche spécifique, et met à jour uniquement
les champs sous sa responsabilité. Grâce aux réducteurs (operator.add), les historiques
de feedback et les journaux d'audit (logs) s'accumulent sans écraser les étapes précédentes.
"""

import sys
import operator
from datetime import datetime, timezone
from typing import TypedDict, Annotated, List, Dict, Any, Optional

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class AgentLogEntry(TypedDict):
    """Représente une trace d'exécution horodatée d'un agent."""

    timestamp: str
    agent: str
    action: str
    details: str


class VerificationState(TypedDict, total=False):
    """
    État global partagé transmis le long du graphe orienté LangGraph.
    Tous les agents communiquent et s'auto-corrigent via cette structure.
    """

    # 1. Affirmation d'entrée et paramétrage
    claim: str  # Affirmation scientifique initiale soumise
    current_query: str  # Requête courante formulée pour la recherche
    max_feedback_iterations: (
        int  # Plafond de réessais (défaut: 2 pour éviter boucles infinies)
    )

    # 2. Analyse sémantique (produit par l'Agent Analyseur)
    extracted_keywords: List[str]  # Mots-clés techniques extraits du claim
    target_categories: List[str]  # Catégories arXiv ciblées (ex: ['cs.CV', 'cs.AI'])
    query_reformulation_notes: (
        str  # Notes sur l'élargissement ou reformulation de la requête
    )

    # 3. Récupération & Reclassement (produit par l'Agent Récupérateur)
    retrieved_passages: List[
        Dict[str, Any]
    ]  # Top passages extraits après recherche hybride et Cross-Encoder
    retrieval_status: str  # 'SUCCESS', 'EMPTY', ou 'LOW_RELEVANCE'
    discarded_passages: int  # Passages écartés sous le seuil de pertinence (dernière récupération)

    # 4. Inférence logique & Consensus (produit par l'Agent Évaluateur)
    evaluated_passages: List[
        Dict[str, Any]
    ]  # Passages enrichis avec postures NLI (SUPPORTS/REFUTES/NEUTRAL)
    verdict_summary: Dict[
        str, Any
    ]  # Résumé du verdict agrégé (global_verdict, compteurs)

    # 5. Contrôle de la Boucle de Rétroaction Active (Active Feedback Loop)
    iteration_count: int  # Nombre de tentatives déjà effectuées (0, 1, 2)
    needs_feedback: bool  # Vrai si les preuves sont jugées insuffisantes pour trancher
    feedback_reason: Optional[str]  # Raison du déclenchement de la rétroaction
    feedback_history: Annotated[
        List[str], operator.add
    ]  # Historique cumulatif des diagnostics de feedback

    # 5 bis. Génération LLM (produit par l'Agent Générateur)
    query_history: List[str]  # Requêtes de recherche déjà essayées (anti-répétition)
    llm_answer: Optional[Dict[str, Any]]  # Réponse rédigée + citations du LLM
    llm_info: Dict[str, Any]  # Statut du LLM : enabled, status, message, latency_s

    # 6. Audit des Citations (produit par l'Agent Vérificateur)
    evidences: List[
        Dict[str, Any]
    ]  # Preuves complètes validées mot pour mot (EvidenceItem)
    citation_audit_passed: (
        bool  # Vrai si toutes les preuves citées existent textuellement
    )

    # 7. Synthèse et Rapport Final (produit par l'Agent Synthèse)
    confidence_breakdown: Optional[
        Dict[str, Any]
    ]  # Décomposition des 4 dimensions du score
    final_report: Optional[
        Dict[str, Any]
    ]  # Rapport officiel complet (ClaimVerificationReport)

    # 8. Traçabilité & Observabilité multi-agents
    agent_logs: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # Journal séquentiel complet des agents


def format_agent_log(agent_name: str, action: str, details: str) -> Dict[str, Any]:
    """
    Crée une entrée de journal d'audit standardisée et horodatée pour un agent.

    Args:
        agent_name: Nom de l'agent (ex: 'Analyseur', 'Récupérateur', etc.).
        action: Verbe ou action effectuée (ex: 'EXTRACTION_MOTS_CLES').
        details: Explication concise du résultat.

    Returns:
        Dictionnaire compatible AgentLogEntry.
    """
    now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    return {
        "timestamp": now_utc,
        "agent": agent_name,
        "action": action,
        "details": details,
    }


def create_initial_state(
    claim: str, max_feedback_iterations: int = 2
) -> VerificationState:
    """
    Initialise l'état du graphe avec les valeurs par défaut sécurisées.

    Args:
        claim: L'affirmation scientifique à vérifier.
        max_feedback_iterations: Nombre maximal de relances de recherche autorisées.

    Returns:
        Objet VerificationState prêt pour être injecté au nœud START de LangGraph.
    """
    init_log = format_agent_log(
        agent_name="Système",
        action="INITIALISATION_GRAPHE",
        details=f"Démarrage de la vérification multi-agents pour : '{claim[:70]}...'",
    )

    return {
        "claim": claim.strip(),
        "current_query": claim.strip(),
        "max_feedback_iterations": max_feedback_iterations,
        "extracted_keywords": [],
        "target_categories": [],
        "query_reformulation_notes": "",
        "retrieved_passages": [],
        "retrieval_status": "PENDING",
        "discarded_passages": 0,
        "evaluated_passages": [],
        "verdict_summary": {},
        "iteration_count": 0,
        "needs_feedback": False,
        "feedback_reason": None,
        "feedback_history": [],
        "query_history": [],
        "llm_answer": None,
        "llm_info": {},
        "evidences": [],
        "citation_audit_passed": False,
        "confidence_breakdown": None,
        "final_report": None,
        "agent_logs": [init_log],
    }
