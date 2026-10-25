"""
Script interactif pour explorer et visualiser la base de données vectorielle (FAISS + Métadonnées).

Exemples d'utilisation :
-------------------------
1. Afficher les statistiques globales de la base :
   python scripts/inspect_db.py --stats

2. Lister les 10 premiers passages indexés :
   python scripts/inspect_db.py --list 10

3. Rechercher des articles par mot-clé dans les titres et contenus :
   python scripts/inspect_db.py --search "transformer"

4. Afficher tous les détails d'un article ou chunk spécifique :
   python scripts/inspect_db.py --view "2610.22370_c0"
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import FAISS_INDEX_FILE, FAISS_METADATA_FILE, VECTOR_DB_DIR
from src.retrieval.vector_store import VectorStoreManager


def load_data():
    if not FAISS_METADATA_FILE.exists():
        print(
            f"[ERREUR] Le fichier de métadonnées {FAISS_METADATA_FILE} est introuvable."
        )
        sys.exit(1)

    with open(FAISS_METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


def show_stats(metadata):
    total_chunks = len(metadata)
    unique_papers = len(set(m["paper_id"] for m in metadata))
    categories = Counter(m["primary_category"] for m in metadata)
    avg_words = (
        sum(len(m["raw_content"].split()) for m in metadata) / total_chunks
        if total_chunks
        else 0
    )

    print("\n" + "=" * 70)
    print("📊 STATISTIQUES DE LA BASE DE DONNÉES VECTORIELLE")
    print("=" * 70)
    print(f"📁 Dossier de stockage   : {VECTOR_DB_DIR}")
    print(
        f"🧠 Index binaire FAISS   : {FAISS_INDEX_FILE.name} ({FAISS_INDEX_FILE.stat().st_size / 1024:.1f} KB)"
    )
    print(
        f"📋 Fichier métadonnées   : {FAISS_METADATA_FILE.name} ({FAISS_METADATA_FILE.stat().st_size / 1024:.1f} KB)"
    )
    print(f"📄 Nombre total de chunks: {total_chunks}")
    print(f"📚 Articles uniques       : {unique_papers}")
    print(f"📝 Longueur moyenne      : {avg_words:.1f} mots / chunk")
    print("\n🏷️ Répartition par catégorie principale :")
    for cat, count in categories.most_common(10):
        bar = "█" * int(count / total_chunks * 30)
        print(f"   {cat:<10} : {count:>3} chunks  {bar}")
    print("=" * 70 + "\n")


def list_items(metadata, limit=10):
    print("\n" + "=" * 70)
    print(f"📋 LISTE DES {min(limit, len(metadata))} PREMIERS CHUNKS")
    print("=" * 70)

    for i, m in enumerate(metadata[:limit], start=1):
        print(
            f"\n[#{i}] ID: {m['chunk_id']} | Papier: {m['paper_id']} | Cat: {m['primary_category']}"
        )
        print(f"   Titre    : {m['title']}")
        authors = ", ".join(m["authors"][:3]) + ("..." if len(m["authors"]) > 3 else "")
        print(f"   Auteurs  : {authors}")
        print(f"   Date     : {m['published_date']}")
        print(f"   Extrait  : \"{m['raw_content'][:130]}...\"")

    print("\n" + "=" * 70)
    print(
        f"💡 Pour voir le texte complet d'un chunk : python scripts/inspect_db.py --view \"{metadata[0]['chunk_id']}\""
    )
    print("=" * 70 + "\n")


def search_items(metadata, keyword):
    keyword_lower = keyword.lower()
    matches = [
        m
        for m in metadata
        if keyword_lower in m["title"].lower()
        or keyword_lower in m["raw_content"].lower()
    ]

    print("\n" + "=" * 70)
    print(f"🔍 RECHERCHE DE '{keyword}' : {len(matches)} passages trouvés")
    print("=" * 70)

    for i, m in enumerate(matches[:8], start=1):
        print(f"\n[{i}] {m['chunk_id']} - {m['title']}")
        print(f"    Extrait : \"{m['raw_content'][:160]}...\"")
        print(f"    Lien    : {m['entry_id']}")

    if len(matches) > 8:
        print(f"\n... et {len(matches) - 8} autres résultats.")
    print("=" * 70 + "\n")


def view_item(metadata, query_id):
    # Chercher par chunk_id ou paper_id
    matches = [
        m for m in metadata if m["chunk_id"] == query_id or m["paper_id"] == query_id
    ]

    if not matches:
        print(f"\n[INFO] Aucun passage trouvé avec l'identifiant '{query_id}'.\n")
        return

    print("\n" + "=" * 70)
    print(f"🔍 DÉTAIL COMPLET DU PASSAGE ({len(matches)} trouvé(s))")
    print("=" * 70)

    for m in matches:
        print(f"\n• Chunk ID        : {m['chunk_id']}")
        print(f"• Papier arXiv    : {m['paper_id']} (Index #{m['chunk_index']})")
        print(f"• Titre           : {m['title']}")
        print(f"• Auteurs         : {', '.join(m['authors'])}")
        print(f"• Date & Catégorie: {m['published_date']} | {m['primary_category']}")
        print(f"• DOI             : {m.get('doi') or 'Non spécifié'}")
        print(f"• URL arXiv       : {m['entry_id']}")
        print(f"• Lien PDF        : {m['pdf_url']}")
        print(
            "\n--- TEXTE BRUT DU PASSAGE (Utilisé pour vérification des citations) ---"
        )
        print(m["raw_content"])
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Inspecteur de la base de données vectorielle."
    )
    parser.add_argument(
        "--stats", action="store_true", help="Afficher les statistiques globales"
    )
    parser.add_argument(
        "--list",
        type=int,
        nargs="?",
        const=10,
        help="Lister N premiers chunks (par défaut 10)",
    )
    parser.add_argument(
        "--search", type=str, help="Chercher par mot-clé dans les textes et titres"
    )
    parser.add_argument(
        "--view", type=str, help="Voir le détail complet d'un chunk ou paper_id"
    )

    args = parser.parse_args()
    metadata = load_data()

    if args.view:
        view_item(metadata, args.view)
    elif args.search:
        search_items(metadata, args.search)
    elif args.list:
        list_items(metadata, limit=args.list)
    else:
        # Par défaut : afficher les statistiques et un échantillon
        show_stats(metadata)
        list_items(metadata, limit=5)


if __name__ == "__main__":
    main()
