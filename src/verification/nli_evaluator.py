"""
Module de Détection de Posture Logique (Stance Detection) avec DeBERTa-v3 NLI.

Rôle dans le projet :
----------------------
Ce module analyse formellement la relation logique entre un passage extrait
d'un article arXiv (la prémisse) et l'affirmation scientifique (l'hypothèse).

Les 3 classes de vérité :
1. SUPPORTS (Entailment) : Le passage démontre ou affirme que l'affirmation est vraie.
2. REFUTES (Contradiction) : Le passage prouve ou affirme que l'affirmation est fausse.
3. NEUTRAL (Neutre / NEI) : Le passage parle du sujet mais ne contient pas de preuve suffisante.

Principe d'Honnêteté Épistémique :
Si les passages sont tous 'NEUTRAL' ou contradictoires entre eux, le système refuse de
forcer une réponse et répond : « Preuves insuffisantes » ou « Sujet débattu ».
"""

import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from src.config import DEFAULT_NLI_MODEL, NLI_LABEL_MAPPING, NLI_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


# ==============================================================================
# SCHÉMA PYDANTIC DU RÉSULTAT NLI
# ==============================================================================


class StanceResult(BaseModel):
    """
    Structure formelle du verdict d'inférence logique pour un passage donné.
    """

    label: str = Field(
        description="Posture dominante : 'SUPPORTS', 'REFUTES' ou 'NEUTRAL'"
    )
    confidence: float = Field(
        description="Probabilité associée au label dominant (entre 0.0 et 1.0)"
    )
    probabilities: Dict[str, float] = Field(
        description="Distribution complète des probabilités pour les 3 classes"
    )
    is_conclusive: bool = Field(
        description="True si la preuve est tranchée (confidence >= seuil et != NEUTRAL)"
    )
    verdict_fr: str = Field(description="Libellé explicatif en français")


# ==============================================================================
# CLASSE PRINCIPALE : NLIEvaluator
# ==============================================================================


class NLIEvaluator:
    """
    Évaluateur d'inférence logique basé sur DeBERTa-v3-small fine-tuné sur NLI.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str = "cpu",
        confidence_threshold: float = NLI_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialise le modèle DeBERTa NLI.

        Args:
            model_name: Modèle NLI HuggingFace (par défaut: 'cross-encoder/nli-deberta-v3-small').
            device: 'cpu' ou 'cuda'.
            confidence_threshold: Seuil de probabilité pour considérer une preuve concluante.
        """
        self.model_name = model_name
        self.device = device
        self.threshold = confidence_threshold

        logger.info(
            f"Chargement du modèle NLI '{self.model_name}' sur {self.device.upper()}..."
        )
        self.model = CrossEncoder(self.model_name, device=self.device)
        logger.info(
            f"Modèle NLI DeBERTa prêt. Seuil de décision : {self.threshold * 100:.0f}%."
        )

    def predict_stance(self, premise: str, hypothesis: str) -> StanceResult:
        """
        Calcule la posture logique d'une prémisse (extrait) par rapport à une hypothèse (claim).

        Args:
            premise: Extrait textuel issu de l'article scientifique.
            hypothesis: L'affirmation scientifique à vérifier.

        Returns:
            Objet StanceResult contenant le label, la confiance et la distribution.
        """
        # Prédiction des logits bruts par DeBERTa
        logits = self.model.predict([(premise, hypothesis)])[0]

        # Calcul des probabilités via fonction Softmax : exp(z_i) / sum(exp(z))
        exp_logits = np.exp(logits - np.max(logits))  # Stabilité numérique
        probs = exp_logits / np.sum(exp_logits)

        # Mapping des probabilités vers les labels normalisés
        prob_dict = {
            "REFUTES": round(float(probs[0]), 4),
            "SUPPORTS": round(float(probs[1]), 4),
            "NEUTRAL": round(float(probs[2]), 4),
        }

        # Détermination du label gagnant
        winning_idx = int(np.argmax(probs))
        raw_label = NLI_LABEL_MAPPING[winning_idx]
        confidence = prob_dict[raw_label]

        # Déterminer si le résultat est concluant
        is_conclusive = (confidence >= self.threshold) and (raw_label != "NEUTRAL")

        # Libellé en français
        if raw_label == "SUPPORTS":
            verdict_fr = f"Soutient l'affirmation ({confidence * 100:.1f}%)"
        elif raw_label == "REFUTES":
            verdict_fr = f"Contredit l'affirmation ({confidence * 100:.1f}%)"
        else:
            verdict_fr = f"Neutre / Ne permet pas de conclure ({confidence * 100:.1f}%)"

        return StanceResult(
            label=raw_label,
            confidence=confidence,
            probabilities=prob_dict,
            is_conclusive=is_conclusive,
            verdict_fr=verdict_fr,
        )

    def evaluate_passages(
        self, claim: str, passages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Évalue la posture de chaque passage retenu par le moteur de recherche.

        Args:
            claim: L'affirmation scientifique.
            passages: Liste des passages (Top-K) issus du Reranker.

        Returns:
            Liste des passages enrichis de leur évaluation NLI ('nli_stance').
        """
        evaluated = []

        for p in passages:
            p_copy = dict(p)
            text_to_eval = p.get("raw_content", p.get("search_content", ""))

            stance = self.predict_stance(premise=text_to_eval, hypothesis=claim)
            p_copy["nli_stance"] = stance.model_dump()
            evaluated.append(p_copy)

        return evaluated

    @staticmethod
    def aggregate_verdict(evaluated_passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Agrège les postures individuelles de tous les passages pour former le
        Verdict Global avec honnêteté épistémique.

        Cas traités :
        - VÉRIFIÉ (SUPPORTS) : Preuves concordantes qui soutiennent le claim.
        - RÉFUTÉ (REFUTES) : Preuves concordantes qui contredisent le claim.
        - DÉBATTU (CONTRADICTION) : Des papiers soutiennent et d'autres contredisent.
        - PREUVES INSUFFISANTES (NEUTRAL) : Pas de preuves claires trouvées.
        """
        if not evaluated_passages:
            return {
                "global_verdict": "PREUVES_INSUFFISANTES",
                "verdict_fr": "Preuves insuffisantes dans le corpus",
                "confidence": 0.0,
                "supports_count": 0,
                "refutes_count": 0,
                "neutral_count": 0,
            }

        supports_scores = []
        refutes_scores = []
        neutral_scores = []

        for p in evaluated_passages:
            nli = p["nli_stance"]
            label = nli["label"]
            conf = nli["confidence"]

            if label == "SUPPORTS" and nli["is_conclusive"]:
                supports_scores.append(conf)
            elif label == "REFUTES" and nli["is_conclusive"]:
                refutes_scores.append(conf)
            else:
                neutral_scores.append(conf)

        n_sup = len(supports_scores)
        n_ref = len(refutes_scores)
        n_neu = len(neutral_scores)

        # Règle d'arbitrage logique :
        if n_sup > 0 and n_ref > 0:
            global_verdict = "SUJET_DEBATTU"
            verdict_fr = "⚠️ Sujet controversé / Débattu dans la littérature (sources contradictoires)"
            avg_conf = (sum(supports_scores) + sum(refutes_scores)) / (n_sup + n_ref)
        elif n_sup > 0 and n_sup >= n_ref:
            global_verdict = "SOUTENU"
            verdict_fr = "✅ Affirmation Scientifiquement Soutenue par la littérature"
            avg_conf = sum(supports_scores) / n_sup
        elif n_ref > 0 and n_ref > n_sup:
            global_verdict = "REFUTE"
            verdict_fr = "❌ Affirmation Scientifiquement Réfutée par la littérature"
            avg_conf = sum(refutes_scores) / n_ref
        else:
            global_verdict = "PREUVES_INSUFFISANTES"
            verdict_fr = "⚪ Preuves insuffisantes pour statuer formellement"
            avg_conf = max(neutral_scores) if neutral_scores else 0.0

        return {
            "global_verdict": global_verdict,
            "verdict_fr": verdict_fr,
            "confidence": round(float(avg_conf), 4),
            "supports_count": n_sup,
            "refutes_count": n_ref,
            "neutral_count": n_neu,
            "total_evaluated": len(evaluated_passages),
        }
