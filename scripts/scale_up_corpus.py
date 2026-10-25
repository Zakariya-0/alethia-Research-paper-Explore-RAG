"""
Script d'Automatisation Globale : Scale-Up du Corpus arXiv à 3 000+ Articles (Phase 6).

Ce script orchestre les 4 étapes indispensables de manière séquentielle et sécurisée :
---------------------------------------------------------------------------------------
1. Collecte ciblée multi-catégories arXiv (cs.LG, cs.AI, cs.CV, cs.CL, cs.RO, cs.CR, cs.SE)
   jusqu'à atteindre le seuil cible (ex: 3 000 articles).
2. Découpage textuel intelligent (Chunking) avec préservation du contexte et nettoyage scientifique.
3. Indexation vectorielle dense FAISS (Meta AI) par lots optimisés sur CPU avec bge-small-en-v1.5.
4. Indexation lexicale sparse BM25Okapi avec tokenisation scientifique spécialisée.
5. Vérification de santé et requête de test sémantique sur la nouvelle base enrichie.

Usage :
-------
python scripts/scale_up_corpus.py [--target 3000] [--batch-size 128]
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List

# Configuration de l'encodage pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import (
    DEFAULT_METADATA_FILE,
    DEFAULT_CHUNKS_FILE,
    FAISS_INDEX_FILE,
    FAISS_METADATA_FILE,
    BM25_INDEX_FILE,
    BM25_METADATA_FILE,
    DEFAULT_ARXIV_CATEGORIES,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from src.data.collector import ArxivCollector
from src.data.chunker import TextChunker, PaperChunk
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_search import HybridRetriever

# Configuration des logs pour un affichage lisible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ScaleUpCorpus")


def count_existing_papers(file_path: Path) -> int:
    """Compte le nombre de lignes dans le fichier JSONL des métadonnées."""
    if not file_path.exists():
        return 0
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def run_collection_step(target_total: int) -> int:
    """
    Étape 1 : Collecte les articles manquants pour atteindre target_total
    en distribuant intelligemment les requêtes sur les sous-domaines clés de l'informatique.
    """
    print("\n" + "=" * 80)
    print("📥 ÉTAPE 1/4 : COLLECTE DES ARTICLES SCIENTIFIQUES ARXIV")
    print("=" * 80)

    current_count = count_existing_papers(DEFAULT_METADATA_FILE)
    needed = target_total - current_count

    print(f"• Articles actuels dans la base : {current_count}")
    print(f"• Objectif cible                : {target_total}")

    if needed <= 0:
        print(
            f"[OK] Le corpus contient déjà {current_count} articles (>= {target_total}). Pas de collecte requise."
        )
        return current_count

    print(f"• Nouveaux articles à collecter : {needed}")
    print("-" * 80)

    # Répartition pondérée des catégories pour couvrir un spectre scientifique équilibré
    # cs.LG (Machine Learning), cs.AI (Intelligence Artificielle), cs.CV (Vision),
    # cs.CL (NLP), cs.RO (Robotique), cs.SE (Génie Logiciel), cs.CR (Cryptographie/Sécurité)
    category_weights = {
        "cs.LG": 0.28,  # ~28% Machine Learning
        "cs.AI": 0.24,  # ~24% Intelligence Artificielle
        "cs.CV": 0.18,  # ~18% Vision par ordinateur
        "cs.CL": 0.18,  # ~18% NLP & LLMs
        "cs.RO": 0.06,  # ~6% Robotique
        "cs.CR": 0.03,  # ~3% Sécurité & Crypto
        "cs.SE": 0.03,  # ~3% Génie logiciel
    }

    collector = ArxivCollector(output_file=DEFAULT_METADATA_FILE)
    total_added = 0

    for cat, weight in category_weights.items():
        if total_added >= needed:
            break

        quota = int(round(needed * weight))
        if quota <= 0:
            quota = 20

        print(
            f"\n[Catégorie {cat}] Téléchargement des métadonnées pour ~{quota} articles..."
        )
        try:
            added_papers = collector.collect_papers(
                categories=[cat],
                max_results=quota,
                download_pdfs=False,  # Pas besoin des PDFs lourds, l'abstract et le titre suffisent
            )
            total_added += len(added_papers)
            print(
                f"[OK] {len(added_papers)} articles ajoutés pour {cat} (Total ajouté: {total_added}/{needed})"
            )
        except Exception as e:
            logger.warning(f"Avertissement lors de la collecte pour {cat}: {e}")
            time.sleep(2)

    final_count = count_existing_papers(DEFAULT_METADATA_FILE)
    print(
        f"\n[BILAN ÉTAPE 1] Total final dans le corpus : {final_count} articles scientifiques."
    )
    return final_count


def run_chunking_step() -> int:
    """
    Étape 2 : Découpage intelligent de tous les articles en chunks avec contexte.
    """
    print("\n" + "=" * 80)
    print("✂️  ÉTAPE 2/4 : DÉCOUPAGE TEXTUEL CONTEXTUALISÉ (CHUNKING)")
    print("=" * 80)

    start_time = time.time()
    chunker = TextChunker(
        chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )

    chunks = chunker.process_metadata_file(
        input_file=DEFAULT_METADATA_FILE, output_file=DEFAULT_CHUNKS_FILE
    )

    elapsed = time.time() - start_time
    total_chunks = len(chunks)
    avg_words = (
        sum(c.word_count for c in chunks) / total_chunks if total_chunks > 0 else 0
    )

    print(f"\n[BILAN ÉTAPE 2] {total_chunks} chunks générés en {elapsed:.2f} secondes.")
    print(f"• Moyenne par chunk : {avg_words:.1f} mots")
    print(f"• Fichier persisté  : {DEFAULT_CHUNKS_FILE}")
    return total_chunks


def run_faiss_indexing_step(batch_size: int = 128) -> int:
    """
    Étape 3 : Ré-indexation complète de FAISS avec encodage par lots vectoriels.
    """
    print("\n" + "=" * 80)
    print(f"⚡ ÉTAPE 3/4 : INDEXATION VECTORIELLE DENSE FAISS (BAAI/bge-small-en-v1.5)")
    print(f"• Batch size CPU : {batch_size}")
    print("=" * 80)

    start_time = time.time()
    vstore = VectorStoreManager()

    # Réinitialisation propre de l'ancien index
    vstore.clear()

    # Chargement de tous les chunks depuis le fichier JSONL
    chunks: List[PaperChunk] = []
    with open(DEFAULT_CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(PaperChunk(**json.loads(line)))

    print(
        f"[INFO] {len(chunks)} chunks chargés pour calcul des embeddings vectoriels..."
    )
    indexed_count = vstore.index_chunks(chunks, batch_size=batch_size)

    elapsed = time.time() - start_time
    rate = indexed_count / elapsed if elapsed > 0 else 0
    print(
        f"\n[BILAN ÉTAPE 3] {indexed_count} vecteurs indexés dans FAISS en {elapsed:.2f} s ({rate:.1f} chunks/sec)."
    )
    print(f"• Fichier binaire FAISS : {FAISS_INDEX_FILE}")
    return indexed_count


def run_bm25_indexing_step() -> int:
    """
    Étape 4 : Tokenisation et construction de l'index inversé BM25Okapi.
    """
    print("\n" + "=" * 80)
    print("🔍 ÉTAPE 4/4 : INDEXATION LEXICALE SPARSE BM25 (BM25Okapi)")
    print("=" * 80)

    start_time = time.time()
    bm25 = BM25Retriever()
    bm25.clear()

    chunks: List[PaperChunk] = []
    with open(DEFAULT_CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(PaperChunk(**json.loads(line)))

    indexed_count = bm25.index_chunks(chunks)
    elapsed = time.time() - start_time

    print(
        f"\n[BILAN ÉTAPE 4] {indexed_count} documents indexés dans BM25 en {elapsed:.2f} secondes."
    )
    print(f"• Fichier BM25 : {BM25_INDEX_FILE}")
    return indexed_count


def run_verification_test(
    sample_query: str = "Video world models retain visual memory of generated content",
):
    """
    Test de santé : exécute une recherche hybride sur la nouvelle base pour valider le bon fonctionnement.
    """
    print("\n" + "=" * 80)
    print("🔬 TEST DE VALIDATION SÉMANTIQUE SUR LA NOUVELLE BASE ÉLARGIE")
    print(f'• Requête de test : "{sample_query}"')
    print("=" * 80)

    vstore = VectorStoreManager()
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(vector_store=vstore, bm25_retriever=bm25)

    results = hybrid.search(sample_query, top_k=3)

    for i, res in enumerate(results, 1):
        print(
            f"\n[RÉSULTAT #{i}] Score RRF: {res['rrf_score']:.5f} ({res['retrieved_by']})"
        )
        print(f"• Titre     : {res['title']}")
        print(
            f"• ID arXiv  : {res['paper_id']} ({res['published_date']}) [{res['primary_category']}]"
        )
        print(f"• Extrait   : \"{res['raw_content'][:200]}...\"")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Scale-Up automatique du corpus scientifique arXiv à 3 000+ articles."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=3000,
        help="Nombre total d'articles cible dans la base (défaut: 3000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Taille des lots pour le calcul des embeddings sur CPU (défaut: 128)",
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Sauter l'étape de collecte et ré-indexer uniquement les données actuelles",
    )

    args = parser.parse_args()

    global_start = time.time()

    print("\n" + "█" * 80)
    print("🚀 PIPELINE DE SCALE-UP DU CORPUS SCIENTIFIQUE (PHASE 6)")
    print("█" * 80)
    print(f"• Seuil cible d'articles : {args.target}")
    print(f"• Batch size vectoriel   : {args.batch_size}")
    print("█" * 80)

    # 1. Collecte
    if not args.skip_collection:
        run_collection_step(target_total=args.target)
    else:
        print("[INFO] Étape de collecte ignorée (--skip-collection).")

    # 2. Chunking
    total_chunks = run_chunking_step()

    # 3. FAISS
    run_faiss_indexing_step(batch_size=args.batch_size)

    # 4. BM25
    run_bm25_indexing_step()

    # 5. Sanity Check
    run_verification_test()

    total_time = time.time() - global_start
    print("\n" + "█" * 80)
    print(f"✅ SCALE-UP RÉUSSI EN {total_time / 60:.1f} MINUTES !")
    print("█" * 80)
    print(f"• Articles indexés : {count_existing_papers(DEFAULT_METADATA_FILE)}")
    print(f"• Chunks produits   : {total_chunks}")
    print(f"• Moteurs prêts    : FAISS IndexFlatIP (384d) + BM25Okapi + Cross-Encoder")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()
