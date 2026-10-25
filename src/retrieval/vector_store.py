"""
Module d'Indexation et de Recherche Vectorielle Dense avec FAISS (Meta AI).

Pourquoi FAISS (Facebook AI Similarity Search) :
------------------------------------------------
1. Spécifié dans la proposition de projet S5 : "Base vectorielle : ChromaDB (ou FAISS)".
2. Performance et Légèreté : FAISS est la référence mondiale de l'industrie pour la recherche
   sémantique haute performance. Développé en C++ par Meta AI, il est ultra-optimisé pour
   le calcul matriciel et la similarité cosinus sur CPU.
3. Persistance simple et fiable : Sauvegarde de l'index binaire sur disque sans dépendre
   d'un serveur de base de données externe ou de compilateurs C++ manquants sous Windows.
"""

import sys
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Encodage sécurisé pour la console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.config import (
    VECTOR_DB_DIR,
    FAISS_INDEX_FILE,
    FAISS_METADATA_FILE,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)
from src.data.chunker import PaperChunk

logger = logging.getLogger(__name__)


# ==============================================================================
# GESTIONNAIRE VECTORIEL FAISS : VectorStoreManager
# ==============================================================================


class VectorStoreManager:
    """
    Gère l'indexation vectorielle dense, la persistance sur disque et la recherche
    sémantique par similarité cosinus avec FAISS et BGE.
    """

    def __init__(
        self,
        persist_dir: Path = VECTOR_DB_DIR,
        index_file: Path = FAISS_INDEX_FILE,
        metadata_file: Path = FAISS_METADATA_FILE,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIMENSION,
        device: str = "cpu",
    ):
        """
        Initialise le gestionnaire vectoriel FAISS.

        Args:
            persist_dir: Dossier de persistance.
            index_file: Chemin du fichier d'index binaire .faiss.
            metadata_file: Chemin du fichier JSON contenant les métadonnées.
            model_name: Modèle d'embeddings dense (BAAI/bge-small-en-v1.5).
            dimension: Dimension des vecteurs (384 pour bge-small).
            device: 'cpu' ou 'cuda'.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = Path(index_file)
        self.metadata_file = Path(metadata_file)
        self.model_name = model_name
        self.dimension = dimension
        self.device = device

        logger.info(
            f"Chargement du modèle d'embeddings '{self.model_name}' sur {self.device.upper()}..."
        )
        self.embedder = SentenceTransformer(self.model_name, device=self.device)

        # Liste stockant les métadonnées de chaque chunk (position i = vecteur i dans FAISS)
        self.metadata: List[Dict[str, Any]] = []

        # Index FAISS basé sur le Produit Scalaire (Inner Product)
        # Avec des vecteurs normalisés L2, le Produit Scalaire est STRICTEMENT ÉGAL à la Similarité Cosinus !
        self.index = faiss.IndexFlatIP(self.dimension)

        # Charger l'index existant s'il est déjà présent sur le disque
        self._load_if_exists()

    def _load_if_exists(self):
        """Charge l'index FAISS et les métadonnées s'ils existent déjà sur disque."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                logger.info(
                    f"Chargement de l'index FAISS existant depuis {self.index_file}..."
                )
                self.index = faiss.read_index(str(self.index_file))

                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

                logger.info(
                    f"Index chargé avec succès : {self.index.ntotal} vecteurs disponibles."
                )
            except Exception as e:
                logger.warning(
                    f"Impossible de charger l'index existant ({e}). Recréation d'un index vide."
                )
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = []

    def save(self):
        """Sauvegarde l'index binaire FAISS et les métadonnées sur le disque."""
        logger.info(
            f"Sauvegarde de l'index FAISS ({self.index.ntotal} vecteurs) vers {self.index_file}..."
        )
        faiss.write_index(self.index, str(self.index_file))

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info("Persistance locale réussie.")

    def count(self) -> int:
        """Retourne le nombre total de passages indexés."""
        return self.index.ntotal

    def index_chunks(self, chunks: List[PaperChunk], batch_size: int = 64) -> int:
        """
        Calcule les embeddings des chunks et les ajoute dans l'index FAISS.

        Args:
            chunks: Liste des PaperChunks à indexer.
            batch_size: Taille de lot pour le calcul des embeddings sur CPU.

        Returns:
            Nombre de chunks ajoutés.
        """
        if not chunks:
            logger.warning("Aucun chunk à indexer.")
            return 0

        total = len(chunks)
        logger.info(
            f"Début de l'indexation de {total} chunks avec {self.model_name}..."
        )

        all_embeddings = []
        new_metadata = []

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.search_content for c in batch]

            # Calcul des embeddings avec normalisation L2 pour la similarité cosinus
            embs = self.embedder.encode(
                texts,
                batch_size=len(batch),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(embs)

            for chunk in batch:
                new_metadata.append(
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
                        "raw_content": chunk.content,  # Utilisé pour la citation mot pour mot
                        "search_content": chunk.search_content,
                    }
                )

            logger.info(
                f"Progression : {min(i + batch_size, total)}/{total} chunks vectorisés."
            )

        # Concaténation de tous les vecteurs en une matrice numpy float32
        embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)

        # Ajout direct dans l'index FAISS
        self.index.add(embeddings_matrix)
        self.metadata.extend(new_metadata)

        # Sauvegarde sur disque
        self.save()

        logger.info(
            f"Indexation terminée. Total de vecteurs dans FAISS : {self.count()}."
        )
        return total

    def search(
        self, query: str, top_k: int = 5, category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche les passages les plus pertinents par similarité cosinus.

        Args:
            query: Affirmation scientifique ou question technique.
            top_k: Nombre de résultats à retourner.
            category_filter: Catégorie arXiv optionnelle (ex: 'cs.AI').

        Returns:
            Liste de dictionnaires ordonnée par score de similarité décroissant.
        """
        if self.count() == 0:
            logger.warning("L'index FAISS est vide. Aucun document à rechercher.")
            return []

        # Instruction officielle du modèle BAAI/bge pour optimiser la recherche de requêtes
        query_text = (
            f"Represent this sentence for searching relevant passages: {query.strip()}"
        )

        query_vector = self.embedder.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        # Si on filtre par catégorie, on demande un peu plus de résultats bruts pour compenser
        fetch_k = min(self.count(), top_k * 4 if category_filter else top_k)

        # Recherche des plus proches voisins dans l'index FAISS
        scores, indices = self.index.search(query_vector, fetch_k)

        results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            # Filtre optionnel sur la catégorie
            if category_filter and meta.get("primary_category") != category_filter:
                continue

            results.append(
                {
                    "chunk_id": meta.get("chunk_id"),
                    "score": round(float(score), 4),  # Similarité cosinus entre 0 et 1
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
        """Réinitialise complètement l'index vectoriel."""
        logger.warning("Réinitialisation de l'index FAISS...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logger.info("Index FAISS réinitialisé.")
