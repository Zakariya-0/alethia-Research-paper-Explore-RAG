"""
Module de Reclassement haute précision par Cross-Encoder (Reranker).

Pourquoi le Reclassement (Reranking) est crucial :
--------------------------------------------------
1. Différence Bi-Encoder vs Cross-Encoder :
   - Bi-Encoder (FAISS) : Encode la requête et le document SÉPARÉMENT en deux vecteurs.
     C'est ultra-rapide (millisecondes), mais les mots de la requête n'interagissent pas
     directement avec les mots du document dans le réseau neuronal.
   - Cross-Encoder (MiniLM) : Passe la requête ET le document SIMULTANÉMENT dans le Transformer.
     Chaque mot de la requête applique un mécanisme d'attention croisée (cross-attention)
     sur chaque mot du document, offrant une précision d'évaluation sémantique inégalée.

2. Architecture de Recherche en Deux Étapes (Two-Stage Retrieval) :
   - Étape 1 (Recherche Hybride rapide) : FAISS + BM25 extraient rapidement 15-20 candidats prometteurs.
   - Étape 2 (Reranker précis) : Le Cross-Encoder réordonne ces 15 candidats et filtre les 3 à 5
     passages ultimes contenant la preuve formelle.
"""

import sys
import logging
from typing import List, Dict, Any, Optional

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from sentence_transformers import CrossEncoder

from src.config import DEFAULT_RERANKER_MODEL, DEFAULT_RERANK_TOP_K

logger = logging.getLogger(__name__)


# ==============================================================================
# CLASSE RERANKER : CrossEncoderReranker
# ==============================================================================


class CrossEncoderReranker:
    """
    Réordonne une liste de passages candidats issus de la recherche hybride
    en évaluant l'attention croisée complète (Cross-Attention) entre requête et texte.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str = "cpu"):
        """
        Initialise le modèle Cross-Encoder.

        Args:
            model_name: Identifiant HuggingFace (par défaut: 'cross-encoder/ms-marco-MiniLM-L-6-v2').
            device: 'cpu' ou 'cuda'.
        """
        self.model_name = model_name
        self.device = device

        logger.info(
            f"Chargement du Cross-Encoder '{self.model_name}' sur {self.device.upper()}..."
        )
        self.model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Cross-Encoder prêt à l'emploi.")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Reclasse les documents candidats en calculant leur score d'attention croisée.

        Args:
            query: Affirmation scientifique ou question technique.
            documents: Liste de dictionnaires candidats issus de FAISS/BM25/RRF.
            top_k: Nombre final de documents à conserver après reclassement.

        Returns:
            Sous-ensemble ordonné des Top-K documents avec 'rerank_score' et métriques.
        """
        if not documents:
            logger.warning("Aucun document à reclasser.")
            return []

        logger.info(
            f"Reclassement Cross-Encoder de {len(documents)} candidats pour la requête : '{query[:60]}...'"
        )

        # 1. Préparation des paires (requête, contenu du document)
        pairs = [
            (query, doc.get("search_content", doc.get("raw_content", "")))
            for doc in documents
        ]

        # 2. Prédiction des scores par le Cross-Encoder (logits bruts)
        raw_scores = self.model.predict(pairs)

        # 3. Normalisation des scores via fonction Sigmoïde pour obtenir une probabilité entre 0.0 et 1.0
        # sigmoide(x) = 1 / (1 + exp(-x))
        normalized_scores = 1.0 / (1.0 + np.exp(-raw_scores))

        # 4. Association des scores et calcul du changement de position
        scored_docs = []
        for orig_rank, (doc, raw_s, norm_s) in enumerate(
            zip(documents, raw_scores, normalized_scores), start=1
        ):
            doc_copy = dict(doc)
            doc_copy["original_hybrid_rank"] = orig_rank
            doc_copy["rerank_raw_score"] = round(float(raw_s), 4)
            doc_copy["rerank_score"] = round(
                float(norm_s), 4
            )  # Score probabilité entre 0.0 et 1.0
            scored_docs.append(doc_copy)

        # 5. Tri par score de reclassement décroissant
        reranked_docs = sorted(
            scored_docs, key=lambda d: d["rerank_score"], reverse=True
        )

        # 6. Mise à jour de l'information de rang final et calcul du delta
        final_top_k = []
        for new_rank, doc in enumerate(reranked_docs[:top_k], start=1):
            doc["final_rank"] = new_rank
            # delta > 0 = document qui a monté dans le classement grâce au Reranker
            doc["rank_delta"] = doc["original_hybrid_rank"] - new_rank
            final_top_k.append(doc)

        logger.info(f"Reclassement terminé. {len(final_top_k)} passages retenus.")
        return final_top_k
