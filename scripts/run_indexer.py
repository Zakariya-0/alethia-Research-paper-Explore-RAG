"""
Script exécutable pour indexer les chunks dans FAISS et tester la recherche sémantique.

Exemples d'utilisation :
-------------------------
1. Indexer tous les chunks traités dans FAISS :
   python scripts/run_indexer.py

2. Indexer ET lancer une recherche sémantique de test :
   python scripts/run_indexer.py --query "Reinforcement learning for multi-agent systems"

3. Tester une recherche sur la base existante sans ré-indexer :
   python scripts/run_indexer.py --query "diffusion models for video generation" --search-only
"""

import sys
import json
import argparse
from pathlib import Path

# Encodage sécurisé pour la console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ajouter la racine du projet au PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import DEFAULT_CHUNKS_FILE, VECTOR_DB_DIR
from src.data.chunker import PaperChunk
from src.retrieval.vector_store import VectorStoreManager


def main():
    parser = argparse.ArgumentParser(
        description="Indexation vectorielle dense avec FAISS et BAAI/bge-small-en-v1.5."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_CHUNKS_FILE),
        help=f"Fichier source des chunks (par défaut: {DEFAULT_CHUNKS_FILE})",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="Language models reasoning and chain of thought",
        help="Requête ou affirmation scientifique de test pour évaluer la recherche",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de résultats pertinents à afficher (par défaut: 3)",
    )

    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Effectuer uniquement la recherche sans ré-indexer les chunks",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Vider complètement la base vectorielle avant d'indexer",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("[START] GESTIONNAIRE VECTORIEL FAISS & RECHERCHE SÉMANTIQUE")
    print("=" * 70)
    print(f"• Répertoire FAISS    : {VECTOR_DB_DIR}")
    print(f"• Fichier source      : {args.input}")
    print("=" * 70 + "\n")

    # Initialisation du Vector Store
    vstore = VectorStoreManager()

    if args.reset:
        print("[INFO] Réinitialisation de l'index...")
        vstore.clear()

    # Phase d'Indexation
    if not args.search_only:
        input_path = Path(args.input)
        if not input_path.exists():
            print(
                f"[ERREUR] Le fichier {input_path} n'existe pas. Veuillez exécuter run_chunker.py d'abord."
            )
            return

        print(f"[INFO] Chargement des chunks depuis {input_path.name}...")
        chunks = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(PaperChunk(**json.loads(line)))

        print(f"[INFO] {len(chunks)} chunks chargés. Calcul des embeddings en cours...")
        vstore.index_chunks(chunks)
        print(
            f"[OK] Indexation terminée. Total dans FAISS : {vstore.count()} chunks.\n"
        )

    # Phase de Recherche de Test
    if args.query:
        print("=" * 70)
        print(f"[RECHERCHE DE TEST] Requête : '{args.query}'")
        print("=" * 70)

        results = vstore.search(args.query, top_k=args.top_k)

        if not results:
            print("[INFO] Aucun résultat pertinent trouvé.")
        else:
            for rank, res in enumerate(results, start=1):
                print(
                    f"\n--- [RANG #{rank}] Similarité Cosinus : {res['score']:.4f} ---"
                )
                print(f"• Titre       : {res['title']}")
                print(
                    f"• Papier ID   : {res['paper_id']} ({res['published_date']}) | Catégorie: {res['primary_category']}"
                )
                authors_str = ", ".join(res["authors"][:4]) + (
                    "..." if len(res["authors"]) > 4 else ""
                )
                print(f"• Auteurs     : {authors_str}")
                print(f"• Lien arXiv  : {res['entry_id']}")
                print(f"• Extrait brut:\n  \"{res['raw_content'][:280]}...\"")

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
