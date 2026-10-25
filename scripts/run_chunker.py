"""
Script exécutable pour lancer le découpage textuel (Chunking) du corpus d'articles.

Exemples d'utilisation :
-------------------------
1. Découpage avec les paramètres optimaux par défaut :
   python scripts/run_chunker.py

2. Découpage avec des tailles de chunks personnalisées :
   python scripts/run_chunker.py --chunk-size 600 --chunk-overlap 100
"""

import sys
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

from src.config import (
    DEFAULT_METADATA_FILE,
    DEFAULT_CHUNKS_FILE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from src.data.chunker import TextChunker


def main():
    parser = argparse.ArgumentParser(
        description="Découpage textuel intelligent des articles arXiv pour l'indexation RAG."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_METADATA_FILE),
        help=f"Fichier source des métadonnées (par défaut: {DEFAULT_METADATA_FILE})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_CHUNKS_FILE),
        help=f"Fichier destination des chunks (par défaut: {DEFAULT_CHUNKS_FILE})",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Taille maximale d'un chunk en caractères (par défaut: {DEFAULT_CHUNK_SIZE})",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chevauchement entre chunks en caractères (par défaut: {DEFAULT_CHUNK_OVERLAP})",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("[START] DÉCOUPAGE TEXTUEL INTELLIGENT (CHUNKING)")
    print("=" * 70)
    print(f"• Fichier source      : {args.input}")
    print(f"• Fichier de sortie   : {args.output}")
    print(f"• Taille des chunks   : {args.chunk_size} caractères (~150-250 mots)")
    print(f"• Chevauchement       : {args.chunk_overlap} caractères")
    print("=" * 70 + "\n")

    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    chunks = chunker.process_metadata_file(
        input_file=Path(args.input), output_file=Path(args.output)
    )

    # Calcul de quelques statistiques intéressantes
    total_chunks = len(chunks)
    avg_words = (
        sum(c.word_count for c in chunks) / total_chunks if total_chunks > 0 else 0
    )
    avg_chars = (
        sum(c.char_count for c in chunks) / total_chunks if total_chunks > 0 else 0
    )

    print("\n" + "=" * 70)
    print(f"[OK] DÉCOUPAGE TERMINÉ AVEC SUCCÈS")
    print(f"• Nombre total de chunks générés : {total_chunks}")
    print(
        f"• Moyenne par chunk             : {avg_words:.1f} mots ({avg_chars:.1f} caractères)"
    )
    print(f"• Fichier généré                : {args.output}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
