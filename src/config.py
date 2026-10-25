"""
Configuration globale du projet S5 : Vérification de claims scientifiques.
Ce fichier centralise les chemins de dossiers, les paramètres de collecte et les constantes.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ==============================================================================
# 1. CHEMINS DU PROJET (Arborescence)
# ==============================================================================

# Racine du projet (dossier Projet-S5)
BASE_DIR = Path(__file__).resolve().parent.parent

# Variables d'environnement : le fichier .env (ignoré par Git) ne remplace jamais
# une variable déjà définie dans le système.
load_dotenv(BASE_DIR / ".env", override=False)

# Dossier principal des données (ignoré par Git pour ne pas surcharger le dépôt)
DATA_DIR = BASE_DIR / "data"

# Sous-dossiers spécifiques
RAW_DATA_DIR = DATA_DIR / "raw"  # Métadonnées brutes en JSON/JSONL
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Chunks de texte nettoyés
PDFS_DIR = DATA_DIR / "pdfs"  # PDFs téléchargés (si besoin)
VECTOR_DB_DIR = DATA_DIR / "chroma_db"  # Persistance de la base vectorielle ChromaDB

# Fichier par défaut où sont stockés les articles collectés
DEFAULT_METADATA_FILE = RAW_DATA_DIR / "arxiv_papers_metadata.jsonl"

# Fichier par défaut où sont stockés les chunks découpés
DEFAULT_CHUNKS_FILE = PROCESSED_DATA_DIR / "arxiv_chunks.jsonl"


# ==============================================================================
# 2. PARAMÈTRES POUR LE CHUNKING (Découpage textuel)
# ==============================================================================

# Taille idéale d'un chunk (en caractères, environ 150 à 250 mots / ~200 tokens)
# Parfaitement adapté au modèle d'embedding bge-small-en-v1.5 (max 512 tokens)
DEFAULT_CHUNK_SIZE = 800

# Chevauchement (overlap) entre deux chunks consécutifs
# Indispensable pour ne pas couper une phrase ou une preuve au milieu
DEFAULT_CHUNK_OVERLAP = 150


# ==============================================================================
# 3. PARAMÈTRES VECTOR STORE & EMBEDDINGS (FAISS / Meta AI)
# ==============================================================================

# Modèle d'embeddings dense (SOTA compact pour l'anglais scientifique)
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Dimension des vecteurs pour bge-small-en-v1.5
EMBEDDING_DIMENSION = 384

# Fichiers de persistance FAISS sur disque
VECTOR_DB_DIR = DATA_DIR / "vector_db"
FAISS_INDEX_FILE = VECTOR_DB_DIR / "index.faiss"
FAISS_METADATA_FILE = VECTOR_DB_DIR / "metadata.json"


# ==============================================================================
# 4. PARAMÈTRES RECHERCHE LEXICALE BM25 & FUSION HYBRIDE (RRF)
# ==============================================================================

# Dossier et fichiers de persistance pour BM25
BM25_DIR = DATA_DIR / "bm25"
BM25_INDEX_FILE = BM25_DIR / "bm25_corpus.pkl"
BM25_METADATA_FILE = BM25_DIR / "bm25_metadata.json"

# Constante k pour la formule Reciprocal Rank Fusion (standard académique: 60)
RRF_K = 60

# Paramètres standards BM25Okapi
BM25_K1 = 1.5  # Saturation des termes
BM25_B = 0.75  # Pénalisation de la longueur du document


# ==============================================================================
# 5. PARAMÈTRES RECLASSEMENT PAR CROSS-ENCODER (RERANKER)
# ==============================================================================

# Modèle Cross-Encoder spécifié dans la proposition de projet S5
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Nombre de passages finaux conservés après reclassement
DEFAULT_RERANK_TOP_K = 3

# Nombre de candidats préliminaires extraits par la recherche hybride
DEFAULT_RERANK_CANDIDATES = 15

# Score minimal du Cross-Encoder (0.0 à 1.0) pour qu'un passage soit retenu.
# En dessous, le passage est écarté avant l'évaluation NLI : un passage hors
# sujet ne doit jamais servir de preuve. Valeur observée sur 100 articles :
# passages hors sujet <= 0.002, passages pertinents >= 0.10 (à recalibrer
# sur un benchmark comme SciFact).
RERANK_MIN_RELEVANCE = 0.05


# ==============================================================================
# 6. PARAMÈTRES INFÉRENCE LOGIQUE NLI (DeBERTa-v3)
# ==============================================================================

# Modèle DeBERTa-v3 fine-tuné sur NLI (MNLI / FEVER / SNLI)
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# Mapping officiel des labels pour cross-encoder/nli-deberta-v3-small
# 0: contradiction, 1: entailment, 2: neutral
NLI_LABEL_MAPPING = {
    0: "REFUTES",  # Contredit l'affirmation
    1: "SUPPORTS",  # Soutient l'affirmation
    2: "NEUTRAL",  # Neutre / Preuves insuffisantes
}

# Seuil de probabilité minimal pour considérer qu'un passage tranche nettement
NLI_CONFIDENCE_THRESHOLD = 0.60


# ==============================================================================
# 7. PONDÉRATIONS POUR LE SCORE DE CONFIANCE COMPOSITE
# ==============================================================================

# Poids du consensus / accord entre sources (35%)
WEIGHT_CONSENSUS = 0.35

# Poids de la qualité sémantique du Cross-Encoder (25%)
WEIGHT_RERANKER = 0.25

# Poids de la force de déduction logique DeBERTa (25%)
WEIGHT_NLI = 0.25

# Poids de la récence des publications scientifiques (15%)
WEIGHT_RECENCY = 0.15

# Décroissance temporelle par an (lambda) pour le calcul de récence
RECENCY_DECAY_RATE = 0.10

# Catégories d'informatique ciblées pour notre corpus (arXiv cs.*)
DEFAULT_ARXIV_CATEGORIES = [
    "cs.AI",  # Intelligence Artificielle générale
    "cs.CL",  # Traitement du Langage Naturel (NLP)
    "cs.LG",  # Machine Learning & Apprentissage profond
    "cs.CV",  # Vision par ordinateur
    "cs.RO",  # Robotique
    "cs.SE",  # Génie logiciel
    "cs.CR",  # Cryptographie et Sécurité
]

# Délai de politesse (en secondes) entre les requêtes vers l'API arXiv
# Recommandation officielle d'arXiv : 3 secondes pour ne pas surcharger leurs serveurs
ARXIV_DELAY_SECONDS = 3.0

# Nombre d'articles par page de requête (taille de lot)
ARXIV_BATCH_SIZE = 100


# ==============================================================================
# 8. PARAMÈTRES DU LLM (génération de la réponse, reformulation des requêtes)
# ==============================================================================
# Tout se règle par variables d'environnement (voir .env.example). Le LLM est
# créé avec langchain.chat_models.init_chat_model : changer LLM_PROVIDER et
# LLM_MODEL suffit pour changer de fournisseur (la clé API se lit dans la
# variable propre au fournisseur, ex. GROQ_API_KEY).


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# Fournisseur et modèle (identifiants tels que les attend init_chat_model)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile").strip()

# Interrupteur global : LLM_ENABLED=false désactive toute génération
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "non",
    "off",
)

# Paramètres d'appel
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.0)
LLM_TIMEOUT_SECONDS = _env_float("LLM_TIMEOUT_SECONDS", 30.0)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 800)

# Langue de la réponse rédigée (le claim peut rester en anglais)
LLM_ANSWER_LANGUAGE = os.getenv("LLM_ANSWER_LANGUAGE", "French").strip()

# Variable d'environnement contenant la clé API, par fournisseur
LLM_API_KEY_VARS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
}


# ==============================================================================
# 3. CRÉATION AUTOMATIQUE DES DOSSIERS
# ==============================================================================


def init_directories():
    """
    Crée automatiquement tous les répertoires nécessaires s'ils n'existent pas encore.
    """
    for folder in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PDFS_DIR, VECTOR_DB_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


# Initialiser les dossiers au chargement de la configuration
init_directories()
