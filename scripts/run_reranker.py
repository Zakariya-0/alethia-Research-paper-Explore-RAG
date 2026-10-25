"""
Script exécutable pour tester le Reclassement haute précision par Cross-Encoder (Reranker).

Ce script démontre l'architecture de recherche en 2 étapes (Two-Stage Retrieval) :
- Étape 1 : Recherche Hybride (FAISS + BM25 via RRF) pour extraire 15 candidats en quelques millisecondes.
- Étape 2 : Reclassement par Cross-Encoder (ms-marco-MiniLM-L-6-v2) pour sélectionner les 3 meilleures preuves.

Exemples d'utilisation :
-------------------------
1. Recherche complète avec reranking :
   python scripts/run_reranker.py --query "diffusion models for video generation"

2. Afficher la comparaison avant/après reclassement :
   python scripts/run_reranker.py --query "autonomous coding agents scaffolding" --show-comparison
"""

import sys
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

from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


def main():
    parser = argparse.ArgumentParser(
        description="Recherche d'information complète en 2 étapes avec Cross-Encoder Reranker."
    )

    parser.add_argument(
        "--query",
        type=str,
        default="diffusion models for video generation",
        help="Affirmation ou requête scientifique à tester",
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=10,
        help="Nombre de candidats préliminaires extraits par la recherche hybride (par défaut: 10)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre final de passages retenus après reclassement (par défaut: 3)",
    )

    parser.add_argument(
        "--show-comparison",
        action="store_true",
        help="Afficher la comparaison avant/après reclassement",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 PIPELINE DE RECHERCHE EN 2 ÉTAPES (HYBRIDE + CROSS-ENCODER RERANKER)")
    print("=" * 80)
    print(f'• Requête             : "{args.query}"')
    print(f"• Candidats Étape 1   : {args.candidates} passages (FAISS + BM25 via RRF)")
    print(f"• Sélection Étape 2   : Top-{args.top_k} passages (Cross-Encoder MiniLM)")
    print("=" * 80 + "\n")

    # Initialisation des moteurs
    hybrid = HybridRetriever()
    reranker = CrossEncoderReranker()

    # 1. Étape 1 : Extraction préliminaire des candidats hybrides
    print(
        f"[ÉTAPE 1] Extraction hybride de {args.candidates} candidats préliminaires..."
    )
    candidates = hybrid.search(args.query, top_k=args.candidates)
    print(f"[OK] {len(candidates)} candidats extraits via RRF.\n")

    # 2. Étape 2 : Reclassement par attention croisée
    print(f"[ÉTAPE 2] Reclassement de précision via Cross-Encoder...")
    final_passages = reranker.rerank(args.query, candidates, top_k=args.top_k)
    print(f"[OK] Reclassement terminé avec succès.\n")

    # Affichage comparatif si demandé
    if args.show_comparison:
        print("=" * 80)
        print("📊 COMPARAISON AVANT / APRÈS RECLASSEMENT PAR LE CROSS-ENCODER")
        print("=" * 80)
        print(
            f"{'Rang Initial (RRF)':<22} | {'Rang Final (Reranker)':<24} | {'Évolution':<10} | Titre du Papier"
        )
        print("-" * 80)
        for p in final_passages:
            init_r = f"Rang #{p['original_hybrid_rank']}"
            fin_r = f"Rang #{p['final_rank']} (Prob: {p['rerank_score']:.1%})"
            delta = p["rank_delta"]
            if delta > 0:
                evol = f"▲ +{delta} (Monté)"
            elif delta < 0:
                evol = f"▼ {delta} (Descendu)"
            else:
                evol = f"= 0 (Stable)"
            print(f"{init_r:<22} | {fin_r:<24} | {evol:<10} | {p['title'][:35]}...")
        print("=" * 80 + "\n")

    # Affichage détaillé des passages finaux retenus pour le LLM
    print("=" * 80)
    print(
        f"🎯 TOP-{len(final_passages)} PASSAGES FINAUX SÉLECTIONNÉS POUR LA VÉRIFICATION"
    )
    print("=" * 80)

    for p in final_passages:
        print(
            f"\n--- [RANG #{p['final_rank']}] Score de Pertinence Cross-Encoder : {p['rerank_score']:.4f} ({p['rerank_score'] * 100:.1f}%) ---"
        )
        print(f"• Titre           : {p['title']}")
        print(
            f"• Papier ID       : {p['paper_id']} ({p['published_date']}) | Catégorie: {p['primary_category']}"
        )
        authors_str = ", ".join(p["authors"][:4]) + (
            "..." if len(p["authors"]) > 4 else ""
        )
        print(f"• Auteurs         : {authors_str}")
        print(
            f"• Découvert par   : {p['retrieved_by']} (Rang Hybride initial: #{p['original_hybrid_rank']})"
        )
        print(f"• Lien arXiv      : {p['entry_id']}")
        print(f"• Extrait textuel exact pour preuve :")
        print(f"  \"{p['raw_content'][:300]}...\"")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
