"""
Script exécutable pour indexer BM25 et tester la Recherche Hybride (FAISS + BM25 avec RRF).

Exemples d'utilisation :
-------------------------
1. Construire l'index BM25 et lancer une recherche hybride :
   python scripts/run_hybrid_search.py --query "diffusion models video generation"

2. Comparer les 3 méthodes côte à côte (Dense seul vs BM25 seul vs Hybride) :
   python scripts/run_hybrid_search.py --query "diffusion models video generation" --compare

3. Ré-indexer complètement BM25 :
   python scripts/run_hybrid_search.py --reindex
"""

import sys
import json
import argparse
from pathlib import Path

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import DEFAULT_CHUNKS_FILE, BM25_DIR
from src.data.chunker import PaperChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.hybrid_search import HybridRetriever


def ensure_bm25_indexed(
    bm25: BM25Retriever, chunks_path: Path, force_reindex: bool = False
):
    """Vérifie que l'index BM25 existe, sinon le construit à partir des chunks traités."""
    if bm25.count() == 0 or force_reindex:
        if not chunks_path.exists():
            print(
                f"[ERREUR] Le fichier {chunks_path} est introuvable. Exécutez run_chunker.py d'abord."
            )
            sys.exit(1)

        print(f"[INFO] Construction de l'index BM25 à partir de {chunks_path.name}...")
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(PaperChunk(**json.loads(line)))

        bm25.index_chunks(chunks)
        print(f"[OK] Index BM25 prêt : {bm25.count()} passages indexés.\n")


def display_comparison(hybrid: HybridRetriever, query: str, top_k: int = 3):
    """Affiche une comparaison claire entre Dense seul, Sparse seul et Hybride RRF."""
    print("\n" + "=" * 80)
    print(f"🔬 ÉTUDE COMPARATIVE DES 3 MODES DE RECHERCHE")
    print(f'• Requête : "{query}"')
    print("=" * 80)

    # 1. Dense Seul (FAISS)
    dense_results = hybrid.vector_store.search(query, top_k=top_k)
    print("\n[MODE 1] RECHERCHE DENSE SEULE (FAISS + Embeddings BGE)")
    print("-" * 80)
    for i, r in enumerate(dense_results, 1):
        print(f"  #{i} | Score Cosinus: {r['score']:.4f} | ID: {r['chunk_id']}")
        print(f"       Titre: {r['title']}")

    # 2. Sparse Seul (BM25)
    sparse_results = hybrid.bm25.search(query, top_k=top_k)
    print("\n[MODE 2] RECHERCHE LEXICALE SEULE (BM25Okapi - Mots-clés exacts)")
    print("-" * 80)
    for i, r in enumerate(sparse_results, 1):
        print(f"  #{i} | Score BM25: {r['score']:>7.2f} | ID: {r['chunk_id']}")
        print(f"       Titre: {r['title']}")

    # 3. Hybride RRF
    hybrid_results = hybrid.search(query, top_k=top_k)
    print("\n[MODE 3] RECHERCHE HYBRIDE (Fusion RRF Dense + Sparse)")
    print("-" * 80)
    for i, r in enumerate(hybrid_results, 1):
        source_badge = f"[{r['retrieved_by']}]"
        print(
            f"  #{i} | Score RRF: {r['rrf_score']:.5f} {source_badge} | ID: {r['chunk_id']}"
        )
        print(f"       Titre: {r['title']}")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Indexation BM25 et Recherche Hybride (FAISS + BM25 via RRF)."
    )

    parser.add_argument(
        "--query",
        type=str,
        default="Speculative forcing autoregressive video generation",
        help="Affirmation ou termes scientifiques à rechercher",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de résultats à afficher (par défaut: 3)",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Afficher la comparaison détaillée (Dense seul vs BM25 seul vs Hybride)",
    )

    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Forcer la ré-indexation complète de BM25",
    )

    args = parser.parse_args()

    # Initialisation des composants
    bm25 = BM25Retriever()
    ensure_bm25_indexed(bm25, Path(DEFAULT_CHUNKS_FILE), force_reindex=args.reindex)

    vstore = VectorStoreManager()
    hybrid = HybridRetriever(vector_store=vstore, bm25_retriever=bm25)

    if args.compare:
        display_comparison(hybrid, args.query, top_k=args.top_k)
    else:
        print("\n" + "=" * 70)
        print(f"🔍 RÉSULTATS DE LA RECHERCHE HYBRIDE (RRF)")
        print(f'• Requête : "{args.query}"')
        print("=" * 70)

        results = hybrid.search(args.query, top_k=args.top_k)

        for rank, res in enumerate(results, start=1):
            print(
                f"\n--- [RANG #{rank}] Score RRF : {res['rrf_score']:.5f} ({res['retrieved_by']}) ---"
            )
            print(f"• Titre       : {res['title']}")
            print(
                f"• Papier ID   : {res['paper_id']} ({res['published_date']}) | Catégorie: {res['primary_category']}"
            )
            authors_str = ", ".join(res["authors"][:3]) + (
                "..." if len(res["authors"]) > 3 else ""
            )
            print(f"• Auteurs     : {authors_str}")
            print(
                f"• Rang Dense  : #{res['dense_rank']} (Cosinus: {res.get('dense_score')})"
            )
            print(
                f"• Rang BM25   : #{res['sparse_rank']} (Score: {res.get('sparse_score')})"
            )
            print(f"• Lien arXiv  : {res['entry_id']}")
            print(f"• Extrait brut:\n  \"{res['raw_content'][:260]}...\"")

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
