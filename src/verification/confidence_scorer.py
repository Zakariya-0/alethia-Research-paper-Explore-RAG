"""
Module de Calcul du Score de Confiance Épistémique Composite.

Rôle dans le projet :
----------------------
Ce module calcule un score de certitude global transparent et explicable pour
chaque affirmation scientifique vérifiée.

Au lieu de s'en remettre à une boîte noire, le score composite intègre 4 dimensions :
1. Consensus scientifique (35%) : Accord ou divergence entre publications indépendantes.
2. Pertinence sémantique (25%) : Qualité de la correspondance évaluée par Cross-Encoder.
3. Certitude logique (25%) : Force de la déduction formelle fournie par DeBERTa-v3.
4. Récence des publications (15%) : Prise en compte de l'obsolescence scientifique
   via une fonction de décroissance exponentielle :
   S_recency = exp(-lambda * age_en_annees)
"""

import re
import sys
import math
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import (
    WEIGHT_CONSENSUS,
    WEIGHT_RERANKER,
    WEIGHT_NLI,
    WEIGHT_RECENCY,
    RECENCY_DECAY_RATE,
)
from src.verification.schemas import ConfidenceBreakdown

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Moteur de calcul mathématique du score de certitude épistémique composite.
    """

    def __init__(
        self,
        weight_consensus: float = WEIGHT_CONSENSUS,
        weight_reranker: float = WEIGHT_RERANKER,
        weight_nli: float = WEIGHT_NLI,
        weight_recency: float = WEIGHT_RECENCY,
        recency_decay_rate: float = RECENCY_DECAY_RATE,
    ):
        """
        Initialise les pondérations du score de confiance composite.

        Args:
            weight_consensus: Poids accordé au consensus multi-sources (par défaut 0.35).
            weight_reranker: Poids accordé à la pertinence du Cross-Encoder (par défaut 0.25).
            weight_nli: Poids accordé à la déduction logique DeBERTa (par défaut 0.25).
            weight_recency: Poids accordé à la récence des articles (par défaut 0.15).
            recency_decay_rate: Taux lambda de décroissance temporelle annuelle (par défaut 0.10).
        """
        # Normalisation défensive des poids pour garantir sum(weights) == 1.0
        total_w = weight_consensus + weight_reranker + weight_nli + weight_recency
        self.w_consensus = weight_consensus / total_w
        self.w_reranker = weight_reranker / total_w
        self.w_nli = weight_nli / total_w
        self.w_recency = weight_recency / total_w
        self.decay_rate = recency_decay_rate

    @staticmethod
    def _extract_pub_year(date_str: Optional[str]) -> Optional[int]:
        """Extrait l'année de publication d'une chaîne de date (ex: '2024-03-15' -> 2024)."""
        if not date_str:
            return None
        match = re.search(r"\b(19\d{2}|20\d{2})\b", str(date_str))
        if match:
            return int(match.group(1))
        return None

    def calculate_recency_score(
        self, evaluated_passages: List[Dict[str, Any]]
    ) -> float:
        """
        Calcule le score de récence temporelle selon la loi de décroissance exponentielle :
        S_recency = (1 / N) * sum(exp(-lambda * age_i))

        Un article récent (année en cours ou précédente) obtient un score proche de 1.0.
        Un article de 5 ans obtient exp(-0.10 * 5) = 0.606.
        """
        if not evaluated_passages:
            return 0.50

        current_year = datetime.now(timezone.utc).year
        recency_scores = []

        # Cibler en priorité les articles concluants
        conclusive_passages = [
            p
            for p in evaluated_passages
            if p.get("nli_stance", {}).get("is_conclusive", False)
        ]
        target_passages = (
            conclusive_passages if conclusive_passages else evaluated_passages
        )

        # Dédupliquer par paper_id pour ne pas sur-pondérer un article découpé en plusieurs chunks
        seen_papers = set()
        for p in target_passages:
            pid = p.get("paper_id", "")
            if pid in seen_papers:
                continue
            seen_papers.add(pid)

            pub_year = self._extract_pub_year(p.get("published_date"))
            if pub_year is not None:
                # Âge en années (>= 0)
                age = max(0, current_year - pub_year)
                # Formule exponentielle : exp(-lambda * age)
                score = math.exp(-self.decay_rate * age)
                recency_scores.append(score)
            else:
                # Score neutre par défaut si date non renseignée
                recency_scores.append(0.75)

        if not recency_scores:
            return 0.50

        return round(float(sum(recency_scores) / len(recency_scores)), 4)

    def calculate_consensus_score(
        self, evaluated_passages: List[Dict[str, Any]], global_verdict: str
    ) -> float:
        """
        Évalue le niveau de consensus scientifique :
        - Accord parfait répliqué sur plusieurs sources indépendantes (>= 2 papiers) : 1.00
        - Accord basé sur une seule source scientifique solide : 0.70
        - Sujet controversé / débat (articles soutenant ET articles réfutant) : pénalité (0.20 à 0.45)
        - Preuves insuffisantes ou neutres : 0.00
        """
        conclusive_supports = {
            p.get("paper_id")
            for p in evaluated_passages
            if p.get("nli_stance", {}).get("label") == "SUPPORTS"
            and p.get("nli_stance", {}).get("is_conclusive", False)
        }
        conclusive_refutes = {
            p.get("paper_id")
            for p in evaluated_passages
            if p.get("nli_stance", {}).get("label") == "REFUTES"
            and p.get("nli_stance", {}).get("is_conclusive", False)
        }

        n_sup = len(conclusive_supports)
        n_ref = len(conclusive_refutes)

        # 1. Cas de controverse active (sources contradictoires)
        if n_sup > 0 and n_ref > 0:
            margin = abs(n_sup - n_ref) / (n_sup + n_ref)
            # Score de consensus réduit (0.20 si égalité parfaite, jusqu'à 0.45 si légère majorité)
            score = 0.20 + (0.25 * margin)
            return round(float(score), 4)

        # 2. Consensus unanime en faveur du soutien
        if n_sup > 0 and n_ref == 0:
            if n_sup >= 2:
                return 1.00  # Multi-sources concordantes
            return 0.70  # Source unique

        # 3. Consensus unanime en faveur de la réfutation
        if n_ref > 0 and n_sup == 0:
            if n_ref >= 2:
                return 1.00  # Multi-sources concordantes
            return 0.70  # Source unique

        # 4. Aucun passage concluant trouvé (prestation neutre / absence de consensus)
        return 0.00

    @staticmethod
    def calculate_reranker_score(evaluated_passages: List[Dict[str, Any]]) -> float:
        """
        Calcule la moyenne des scores de pertinence sémantique du Cross-Encoder.
        """
        if not evaluated_passages:
            return 0.0

        conclusive = [
            p
            for p in evaluated_passages
            if p.get("nli_stance", {}).get("is_conclusive", False)
        ]
        target = conclusive if conclusive else evaluated_passages

        scores = [float(p.get("rerank_score", 0.0)) for p in target]
        return round(float(sum(scores) / len(scores)), 4)

    @staticmethod
    def calculate_nli_score(evaluated_passages: List[Dict[str, Any]]) -> float:
        """
        Calcule la moyenne de confiance logique fournie par DeBERTa-v3 sur les preuves.
        """
        if not evaluated_passages:
            return 0.0

        conclusive = [
            p
            for p in evaluated_passages
            if p.get("nli_stance", {}).get("is_conclusive", False)
        ]

        if conclusive:
            confidences = [float(p["nli_stance"]["confidence"]) for p in conclusive]
            return round(float(sum(confidences) / len(confidences)), 4)

        # Si aucune preuve n'est concluante, la certitude sur le claim est faible
        neutral_confs = [
            float(p.get("nli_stance", {}).get("confidence", 0.0))
            for p in evaluated_passages
        ]
        # Pénalisation car le modèle n'a pas pu trancher
        avg_neutral = sum(neutral_confs) / len(neutral_confs) if neutral_confs else 0.0
        return round(float(avg_neutral * 0.40), 4)

    def compute_confidence(
        self, evaluated_passages: List[Dict[str, Any]], global_verdict: str
    ) -> ConfidenceBreakdown:
        """
        Synthétise les 4 dimensions pour produire le ConfidenceBreakdown complet.

        Args:
            evaluated_passages: Liste des passages avec 'rerank_score' et 'nli_stance'.
            global_verdict: Verdict global issu de NLIEvaluator ('SOUTENU', 'REFUTE', etc.).

        Returns:
            Objet ConfidenceBreakdown validé par Pydantic.
        """
        s_consensus = self.calculate_consensus_score(evaluated_passages, global_verdict)
        s_reranker = self.calculate_reranker_score(evaluated_passages)
        s_nli = self.calculate_nli_score(evaluated_passages)
        s_recency = self.calculate_recency_score(evaluated_passages)

        # Calcul synthétique pondéré
        composite = (
            self.w_consensus * s_consensus
            + self.w_reranker * s_reranker
            + self.w_nli * s_nli
            + self.w_recency * s_recency
        )
        composite = max(0.0, min(1.0, composite))
        composite = round(float(composite), 4)

        # Génération d'une explication textuelle limpide
        weights_dict = {
            "consensus": round(self.w_consensus, 2),
            "reranker": round(self.w_reranker, 2),
            "nli": round(self.w_nli, 2),
            "recency": round(self.w_recency, 2),
        }

        explanation_parts = []
        if s_consensus >= 0.90:
            explanation_parts.append(
                "consensus scientifique solide et répliqué sur plusieurs articles indépendants"
            )
        elif s_consensus >= 0.65:
            explanation_parts.append(
                "preuve appuyée par une source unique solide mais sans réplication indépendante"
            )
        elif global_verdict == "SUJET_DEBATTU":
            explanation_parts.append(
                "controverse active dans la littérature avec des sources contradictoires"
            )
        else:
            explanation_parts.append("absence de consensus tranché dans le corpus")

        explanation_parts.append(
            f"pertinence sémantique Cross-Encoder de {s_reranker * 100:.1f}%"
        )
        explanation_parts.append(f"certitude logique DeBERTa de {s_nli * 100:.1f}%")
        explanation_parts.append(
            f"indice de récence temporelle de {s_recency * 100:.1f}%"
        )

        explanation = (
            f"Score composite de {composite * 100:.1f}% justifié par : "
            + ", ".join(explanation_parts)
            + "."
        )

        return ConfidenceBreakdown(
            score_consensus=s_consensus,
            score_reranker=s_reranker,
            score_nli=s_nli,
            score_recency=s_recency,
            weights=weights_dict,
            composite_score=composite,
            explanation=explanation,
        )
