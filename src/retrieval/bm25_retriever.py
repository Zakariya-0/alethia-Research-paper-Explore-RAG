"""
Module d'Indexation et de Recherche Lexicale avec BM25 (Best Matching 25).

Pourquoi BM25 :
----------------
1. Correspondance exacte : Capture les termes rares, acronymes (LoRA, DeBERTa, BERT, BLEU),
   noms propres d'auteurs et versions de modèles que les embeddings vectoriels ont tendance à lisser.
2. Formule mathématique (BM25Okapi) :
   - Term Frequency (TF) avec saturation k1 : évite qu'un mot répété 50 fois fausse la pertinence.
   - Inverse Document Frequency (IDF) : accorde beaucoup plus de poids aux mots rares qu'aux mots fréquents.
   - Pénalisation de longueur b : normalise le score pour ne pas favoriser artificiellement les longs documents.
"""

import re
import sys
import json
import pickle
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
from rank_bm25 import BM25Okapi

from src.config import BM25_DIR, BM25_INDEX_FILE, BM25_METADATA_FILE, BM25_K1, BM25_B
from src.data.chunker import PaperChunk

logger = logging.getLogger(__name__)

# Stopwords scientifiques anglais courants (évite de polluer l'index avec des mots vides)
STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "its",
    "itself",
    "more",
    "most",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
}


# ==============================================================================
# TOKENISEUR SCIENTIFIQUE
# ==============================================================================


def tokenize_scientific(text: str) -> List[str]:
    """
    Tokenise un texte scientifique pour BM25 :
    - Convertit en minuscules.
    - Préserve les acronymes, noms composés et versions (ex: 'llama-3.1', 'resnet-50', 'cs.ai').
    - Élimine les stopwords anglais et les signes de ponctuation isolés.
    """
    if not text:
        return []

    text_lower = text.lower()

    # Regex capturant :
    # 1. Mots avec tirets ou points internes (ex: llama-3.1, cs.ai, v1.5)
    # 2. Mots alphanumériques standards
    raw_tokens = re.findall(r"[a-zA-Z0-9]+(?:[-._][a-zA-Z0-9]+)*", text_lower)

    # Filtrer les stopwords et les tokens d'un seul caractère non significatif
    tokens = [t for t in raw_tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


# ==============================================================================
# GESTIONNAIRE BM25 : BM25Retriever
# ==============================================================================


class BM25Retriever:
    """
    Moteur de recherche lexicale utilisant l'algorithme BM25Okapi avec persistance locale.
    """

    def __init__(
        self,
        persist_dir: Path = BM25_DIR,
        index_file: Path = BM25_INDEX_FILE,
        metadata_file: Path = BM25_METADATA_FILE,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = Path(index_file)
        self.metadata_file = Path(metadata_file)
        self.k1 = k1
        self.b = b

        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []
        self.metadata: List[Dict[str, Any]] = []

        # Charger l'index existant s'il est déjà disponible
        self._load_if_exists()

    def _load_if_exists(self):
        """Charge l'index BM25 et les métadonnées depuis le disque."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                logger.info(
                    f"Chargement de l'index BM25 existant depuis {self.index_file}..."
                )
                with open(self.index_file, "rb") as f:
                    self.corpus_tokens = pickle.load(f)

                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

                # Reconstruire l'objet BM25 à partir des tokens chargés
                self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
                logger.info(
                    f"Index BM25 chargé avec succès : {len(self.metadata)} documents prêts."
                )
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de l'index BM25 ({e}).")
                self.bm25 = None
                self.corpus_tokens = []
                self.metadata = []

    def save(self):
        """Sauvegarde les tokens tokenisés du corpus et les métadonnées sur le disque."""
        logger.info(
            f"Sauvegarde de l'index BM25 ({len(self.metadata)} documents) vers {self.persist_dir}..."
        )
        with open(self.index_file, "wb") as f:
            pickle.dump(self.corpus_tokens, f)

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info("Persistance BM25 locale réussie.")

    def count(self) -> int:
        """Retourne le nombre total de documents indexés."""
        return len(self.metadata)

    def index_chunks(self, chunks: List[PaperChunk]) -> int:
        """
        Tokenise les chunks et construit l'index inversé BM25.

        Args:
            chunks: Liste des PaperChunks à indexer.

        Returns:
            Nombre de chunks indexés.
        """
        if not chunks:
            logger.warning("Aucun chunk à indexer pour BM25.")
            return 0

        total = len(chunks)
        logger.info(f"Tokenisation et indexation BM25 de {total} chunks...")

        self.corpus_tokens = []
        self.metadata = []

        for chunk in chunks:
            # Tokeniser le texte contextualisé (titre + extrait)
            tokens = tokenize_scientific(chunk.search_content)
            self.corpus_tokens.append(tokens)

            self.metadata.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.title,
                    "authors": chunk.authors,
                    "published_date": chunk.published_date,
                    "primary_category": chunk.primary_category,
                    "doi": chunk.doi,
                    "entry_id": chunk.entry_id,
                    "pdf_url": chunk.pdf_url,
                    "raw_content": chunk.content,
                    "search_content": chunk.search_content,
                }
            )

        # Création du modèle BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)

        # Sauvegarde sur disque
        self.save()

        logger.info(f"Indexation BM25 terminée. {self.count()} chunks indexés.")
        return total

    def search(
        self, query: str, top_k: int = 5, category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche les passages les plus pertinents selon le score BM25.

        Args:
            query: Requête textuelle ou affirmation.
            top_k: Nombre de résultats à retourner.
            category_filter: Filtre optionnel sur la catégorie arXiv.

        Returns:
            Liste de dictionnaires ordonnée par score BM25 décroissant.
        """
        if self.bm25 is None or self.count() == 0:
            logger.warning("L'index BM25 est vide. Aucun document à rechercher.")
            return []

        query_tokens = tokenize_scientific(query)
        if not query_tokens:
            logger.warning(f"Aucun token valide extrait de la requête : '{query}'")
            return []

        # Calcul des scores BM25 pour tous les documents du corpus
        raw_scores = self.bm25.get_scores(query_tokens)

        # Trier les indices par score décroissant
        ranked_indices = np.argsort(raw_scores)[::-1]

        results: List[Dict[str, Any]] = []

        for idx in ranked_indices:
            score = float(raw_scores[idx])
            # Si le score est nul, les documents suivants n'ont aucun mot en commun
            if score <= 0.0:
                break

            meta = self.metadata[idx]

            # Filtre optionnel par catégorie
            if category_filter and meta.get("primary_category") != category_filter:
                continue

            results.append(
                {
                    "chunk_id": meta.get("chunk_id"),
                    "score": round(
                        score, 4
                    ),  # Score brut BM25 (plus c'est élevé, plus c'est pertinent)
                    "paper_id": meta.get("paper_id"),
                    "title": meta.get("title"),
                    "authors": meta.get("authors", []),
                    "published_date": meta.get("published_date"),
                    "primary_category": meta.get("primary_category"),
                    "doi": meta.get("doi"),
                    "pdf_url": meta.get("pdf_url"),
                    "entry_id": meta.get("entry_id"),
                    "raw_content": meta.get("raw_content"),
                    "search_content": meta.get("search_content"),
                }
            )

            if len(results) >= top_k:
                break

        return results

    def clear(self):
        """Réinitialise l'index BM25."""
        logger.warning("Réinitialisation de l'index BM25...")
        self.bm25 = None
        self.corpus_tokens = []
        self.metadata = []
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logger.info("Index BM25 réinitialisé.")
