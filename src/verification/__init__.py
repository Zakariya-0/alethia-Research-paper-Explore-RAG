"""
Module verification : Inférence Logique (DeBERTa NLI), Vérification Textuelle Déterministe
des Citations et Calcul du Score de Confiance Épistémique Composite.
"""

from .nli_evaluator import NLIEvaluator, StanceResult
from .citation_verifier import CitationVerifier, CitationCheckResult
from .confidence_scorer import ConfidenceScorer
from .schemas import EvidenceItem, ConfidenceBreakdown, ClaimVerificationReport

__all__ = [
    "NLIEvaluator",
    "StanceResult",
    "CitationVerifier",
    "CitationCheckResult",
    "ConfidenceScorer",
    "EvidenceItem",
    "ConfidenceBreakdown",
    "ClaimVerificationReport",
]
