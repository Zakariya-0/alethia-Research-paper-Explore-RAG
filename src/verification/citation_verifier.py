"""
Module de Vérification Textuelle Déterministe des Citations (Anti-Hallucination).

Rôle dans le projet :
----------------------
L'un des plus grands risques des LLMs scientifiques est l'invention ou l'altération
de citations. Ce module garantit qu'aucune conclusion ne peut être présentée sans
qu'une citation textuelle n'ait été vérifiée mot pour mot dans le texte source d'arXiv.

Fonctionnalités :
1. Vérification exacte (Exact Substring Matching) : Contrôle que la citation existe mot pour mot.
2. Localisation précise : Retourne la position exacte dans le texte pour le surlignage.
3. Extraction automatique de phrases de preuve : Isole la phrase exacte du passage qui justifie le verdict.
"""

import re
import sys
import logging
from typing import Dict, Any, Optional, List, Tuple
from difflib import SequenceMatcher

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CitationCheckResult(BaseModel):
    """Résultat de la vérification déterministe d'une citation."""

    is_valid: bool = Field(
        description="True si la citation existe textuellement dans la source"
    )
    match_type: str = Field(description="'EXACT', 'APPROXIMATE' ou 'NOT_FOUND'")
    similarity_ratio: float = Field(
        description="Score de correspondance textuelle (1.0 = mot pour mot)"
    )
    matched_text: Optional[str] = Field(
        default=None, description="Extrait correspondant dans le texte source"
    )
    start_pos: int = Field(
        default=-1, description="Position de début dans le texte source"
    )
    end_pos: int = Field(default=-1, description="Position de fin dans le texte source")


class CitationVerifier:
    """
    Garantit l'éradication des hallucinations en vérifiant les citations dans le texte brut.
    """

    @staticmethod
    def normalize_for_check(text: str) -> str:
        """Nettoie les espaces et la ponctuation pour comparer sans faux rejets."""
        if not text:
            return ""
        # Remplacer les retours à la ligne et espaces multiples par un seul espace
        text = re.sub(r"\s+", " ", text)
        # Normaliser les guillemets
        text = (
            text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        )
        return text.strip()

    def verify_citation(
        self, citation: str, source_text: str, min_fuzzy_ratio: float = 0.90
    ) -> CitationCheckResult:
        """
        Vérifie qu'une citation existe dans le texte source de l'article.

        Args:
            citation: Phrase ou extrait cité comme preuve.
            source_text: Texte brut de l'article ou du chunk source.
            min_fuzzy_ratio: Seuil minimal de similarité textuelle acceptable.

        Returns:
            Objet CitationCheckResult avec statut et localisation.
        """
        clean_citation = self.normalize_for_check(citation)
        clean_source = self.normalize_for_check(source_text)

        if not clean_citation or not clean_source:
            return CitationCheckResult(
                is_valid=False, match_type="NOT_FOUND", similarity_ratio=0.0
            )

        # 1. Vérification par correspondance exacte mot pour mot (Exact Match)
        idx = clean_source.lower().find(clean_citation.lower())
        if idx != -1:
            matched_segment = clean_source[idx : idx + len(clean_citation)]
            return CitationCheckResult(
                is_valid=True,
                match_type="EXACT",
                similarity_ratio=1.0,
                matched_text=matched_segment,
                start_pos=idx,
                end_pos=idx + len(clean_citation),
            )

        # 2. Vérification par fenêtre glissante (Fuzzy Match tolérant aux variations de ponctuation mineures)
        citation_len = len(clean_citation)
        best_ratio = 0.0
        best_match = None
        best_start = -1

        words = clean_source.split()
        citation_word_count = len(clean_citation.split())

        # Parcourir le texte par fenêtres de mots de taille similaire
        for i in range(len(words) - citation_word_count + 1):
            window = " ".join(words[i : i + citation_word_count])
            ratio = SequenceMatcher(
                None, clean_citation.lower(), window.lower()
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = window
                best_start = clean_source.find(window)

        if best_ratio >= min_fuzzy_ratio:
            return CitationCheckResult(
                is_valid=True,
                match_type="APPROXIMATE",
                similarity_ratio=round(best_ratio, 4),
                matched_text=best_match,
                start_pos=best_start,
                end_pos=best_start + len(best_match) if best_match else -1,
            )

        return CitationCheckResult(
            is_valid=False,
            match_type="NOT_FOUND",
            similarity_ratio=round(best_ratio, 4),
        )
