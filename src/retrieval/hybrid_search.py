"""
Module de Recherche Hybride combinant Dense (FAISS) et Sparse (BM25) via RRF.

Algorithme Reciprocal Rank Fusion (RRF) :
------------------------------------------
Les scores de similarité cosinus (FAISS, bornés entre 0 et 1) et les scores BM25
(non bornés, de 0 à +l'infini) ne peuvent pas être simplement additionnés sans biais.

RRF résout ce problème de manière élégante et éprouvée en se basant sur les RANGS :
    RRF_Score(d) = w_dense * (1 / (k + rank_dense(d))) + w_sparse * (1 / (k + rank_sparse(d)))

Où :
- k est une constante de lissage (standard académique = 60).
- rank(d) est la position du document (1, 2, 3...).
- Si un document est absent du Top-N d'un moteur, son terme vaut 0.
- Un document présent dans les deux classements reçoit un boost naturel massif.
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

from src.config import RRF_K, DEFAULT_RERANK_TOP_K, DEFAULT_RERANK_CANDIDATES
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


# ==============================================================================
# MOTEUR DE RECHERCHE HYBRIDE : HybridRetriever
# ==============================================================================


class HybridRetriever:
    """
    Combine les résultats de FAISS (dense / sémantique) et de BM25 (sparse / lexical)
    en un classement unifié via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = RRF_K,
    ):
        """
        Initialise le chercheur hybride.
        Charge automatiquement les deux index existants si non fournis.
        """
        self.vector_store = vector_store or VectorStoreManager()
        self.bm25 = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k

        logger.info(
            f"HybridRetriever initialisé : FAISS ({self.vector_store.count()} chunks) | "
            f"BM25 ({self.bm25.count()} chunks) | RRF k={self.rrf_k}"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        candidate_multiplier: int = 4,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Exécute la recherche hybride RRF.

        Args:
            query: Affirmation scientifique ou question technique.
            top_k: Nombre final de résultats à retourner.
            dense_weight: Poids accordé à la recherche sémantique vectorielle (FAISS).
            sparse_weight: Poids accordé à la recherche lexicale par mots-clés (BM25).
            candidate_multiplier: Multiplicateur pour collecter les candidats préliminaires.
            category_filter: Filtre optionnel sur la catégorie arXiv.

        Returns:
            Liste fusionnée et ordonnée par RRF score décroissant.
        """
        num_candidates = max(top_k * candidate_multiplier, 20)

        # 1. Récupération des candidats denses (FAISS)
        dense_results = self.vector_store.search(
            query=query, top_k=num_candidates, category_filter=category_filter
        )

        # 2. Récupération des candidats sparses (BM25)
        sparse_results = self.bm25.search(
            query=query, top_k=num_candidates, category_filter=category_filter
        )

        # 3. Dictionnaire pour accumuler les scores RRF par chunk_id
        doc_pool: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        # Intégration du classement Dense (FAISS)
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            doc_pool[cid] = item
            rrf_contribution = dense_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contribution

            # Sauvegarder les détails du composant
            item["dense_rank"] = rank
            item["dense_score"] = item["score"]

        # Intégration du classement Sparse (BM25)
        for rank, item in enumerate(sparse_results, start=1):
            cid = item["chunk_id"]
            if cid not in doc_pool:
                doc_pool[cid] = item

            rrf_contribution = sparse_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contribution

            # Sauvegarder les détails du composant
            doc_pool[cid]["sparse_rank"] = rank
            doc_pool[cid]["sparse_score"] = item["score"]

        # 4. Tri de tous les candidats par score RRF décroissant
        sorted_chunk_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        # 5. Formatage final des Top-K résultats
        final_results: List[Dict[str, Any]] = []

        for cid in sorted_chunk_ids[:top_k]:
            doc_data = doc_pool[cid]
            has_dense = "dense_rank" in doc_data
            has_sparse = "sparse_rank" in doc_data

            if has_dense and has_sparse:
                retrieved_by = "both (Dense + BM25)"
            elif has_dense:
                retrieved_by = "dense_only (FAISS)"
            else:
                retrieved_by = "sparse_only (BM25)"

            final_results.append(
                {
                    "chunk_id": cid,
                    "rrf_score": round(rrf_scores[cid], 5),
                    "retrieved_by": retrieved_by,
                    "dense_rank": doc_data.get("dense_rank"),
                    "dense_score": doc_data.get("dense_score"),
                    "sparse_rank": doc_data.get("sparse_rank"),
                    "sparse_score": doc_data.get("sparse_score"),
                    "paper_id": doc_data.get("paper_id"),
                    "title": doc_data.get("title"),
                    "authors": doc_data.get("authors", []),
                    "published_date": doc_data.get("published_date"),
                    "primary_category": doc_data.get("primary_category"),
                    "doi": doc_data.get("doi"),
                    "pdf_url": doc_data.get("pdf_url"),
                    "entry_id": doc_data.get("entry_id"),
                    "raw_content": doc_data.get("raw_content"),
                    "search_content": doc_data.get("search_content"),
                }
            )

        return final_results

    def search_and_rerank(
        self,
        query: str,
        top_k: int = DEFAULT_RERANK_TOP_K,
        num_candidates: int = DEFAULT_RERANK_CANDIDATES,
        reranker: Optional[CrossEncoderReranker] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pipeline complet en Deux Étapes (Two-Stage Retrieval) :
        1. Recherche Hybride (FAISS + BM25 via RRF) pour extraire `num_candidates` passages.
        2. Reclassement par Cross-Encoder (MiniLM) pour sélectionner les `top_k` passages finaux.

        Args:
            query: Requête ou affirmation scientifique.
            top_k: Nombre final de passages retournés.
            num_candidates: Nombre de candidats pré-sélectionnés par RRF.
            reranker: Instance optionnelle de CrossEncoderReranker.
            category_filter: Filtre optionnel sur la catégorie arXiv.

        Returns:
            Les passages ultimes classés par pertinence Cross-Encoder.
        """
        # Étape 1 : Recherche hybride de candidats préliminaires
        candidates = self.search(
            query=query, top_k=num_candidates, category_filter=category_filter
        )

        if not candidates:
            return []

        # Initialiser le reranker si non fourni
        rerank_engine = reranker or CrossEncoderReranker()

        # Étape 2 : Reclassement de précision
        reranked_results = rerank_engine.rerank(
            query=query, documents=candidates, top_k=top_k
        )

        return reranked_results
