"""
Script exécutable pour lancer la collecte d'articles scientifiques arXiv.

Exemples d'utilisation :
-------------------------
1. Collecte rapide de 20 articles récents en IA et NLP :
   python scripts/run_collector.py --limit 20

2. Collecte ciblée sur des catégories spécifiques :
   python scripts/run_collector.py --categories cs.AI cs.LG --limit 50

3. Collecte avec des mots-clés spécifiques (ex: transformers, fact-checking) :
   python scripts/run_collector.py --keywords "transformers" "hallucination" --limit 15

4. Collecte avec téléchargement des fichiers PDF en local :
   python scripts/run_collector.py --limit 5 --download-pdfs
"""

import sys
import argparse
from pathlib import Path

# Configuration de l'encodage pour Windows (évite les erreurs UnicodeEncodeError sur cp1252)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ajouter la racine du projet au PYTHONPATH pour pouvoir importer 'src'
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data.collector import ArxivCollector
from src.config import DEFAULT_ARXIV_CATEGORIES, DEFAULT_METADATA_FILE


def main():
    parser = argparse.ArgumentParser(
        description="Outil de collecte d'articles scientifiques arXiv pour le projet S5."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Nombre maximal d'articles à collecter (par défaut : 25).",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help=f"Catégories arXiv ciblées (par défaut : {' '.join(DEFAULT_ARXIV_CATEGORIES)}).",
    )

    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Mots-clés optionnels à rechercher dans le titre et le résumé.",
    )

    parser.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Si activé, télécharge également les fichiers PDF dans data/pdfs/.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("[START] DÉMARRAGE DE LA COLLECTE D'ARTICLES SCIENTIFIQUES (ARXIV)")
    print("=" * 70)
    print(f"• Nombre demandé      : {args.limit}")
    print(f"• Catégories          : {args.categories or DEFAULT_ARXIV_CATEGORIES}")
    print(f"• Mots-clés           : {args.keywords or 'Aucun (tous les articles)'}")
    print(
        f"• Téléchargement PDF  : {'Activé' if args.download_pdfs else 'Désactivé (Métadonnées & abstracts seuls)'}"
    )
    print(f"• Fichier de sortie   : {DEFAULT_METADATA_FILE}")
    print("=" * 70 + "\n")

    collector = ArxivCollector()
    papers = collector.collect_papers(
        categories=args.categories,
        keywords=args.keywords,
        max_results=args.limit,
        download_pdfs=args.download_pdfs,
    )

    print("\n" + "=" * 70)
    print(
        f"[OK] COLLECTE TERMINÉE : {len(papers)} nouveaux articles enregistrés avec succès."
    )
    print(f"[INFO] Fichier consultable : {DEFAULT_METADATA_FILE}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
