"""
Module de découpage textuel intelligent (Chunking) pour articles scientifiques.

Pourquoi le Chunking est indispensable en RAG :
1. Limite de contexte des modèles d'embeddings :
   Un modèle comme 'bge-small-en-v1.5' accepte au maximum 512 tokens.
   Envoyer un article entier dépasserait cette limite ou noierait les détails précis.
2. Précision de la recherche :
   Retrouver un passage précis (ex: 150-250 mots) permet d'isoler l'argument exact
   qui soutient ou réfute une affirmation scientifique.
3. Continuité sémantique (Overlap) :
   Un chevauchement (ex: 150 caractères) entre deux chunks consécutifs garantit
   qu'une phrase ou une preuve scientifique ne soit pas coupée au milieu.
"""

import re
import sys
import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNKS_FILE,
    DEFAULT_METADATA_FILE,
)
from src.data.collector import PaperMetadata

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. MODÈLE DE DONNÉES D'UN CHUNK (Pydantic)
# ==============================================================================


class PaperChunk(BaseModel):
    """
    Représente un fragment textuel (chunk) enrichi de toutes les métadonnées
    de son article d'origine.
    """

    chunk_id: str = Field(
        description="Identifiant unique du chunk (ex: '2401.12345_c0')"
    )
    paper_id: str = Field(description="Identifiant arXiv de l'article parent")
    chunk_index: int = Field(
        description="Position du chunk dans l'article (0, 1, 2...)"
    )
    total_chunks: int = Field(
        default=1, description="Nombre total de chunks issus de cet article"
    )

    # Texte brut exact pour la citation mot pour mot
    content: str = Field(description="Extrait textuel brut issu de l'article")

    # Texte enrichi pour la recherche (inclut le titre pour donner du contexte au vecteur)
    search_content: str = Field(
        description="Texte contextualisé optimisé pour les embeddings et BM25"
    )

    # Métadonnées de l'article parent conservées pour la traçabilité
    title: str = Field(description="Titre de l'article")
    authors: List[str] = Field(default_factory=list, description="Auteurs de l'article")
    published_date: str = Field(description="Date de publication (AAAA-MM-JJ)")
    primary_category: str = Field(description="Catégorie principale arXiv")
    doi: Optional[str] = Field(default=None, description="DOI officiel")
    entry_id: str = Field(description="Lien de la page arXiv")
    pdf_url: str = Field(description="Lien direct vers le PDF")

    # Métriques du chunk
    char_count: int = Field(description="Nombre de caractères dans le chunk")
    word_count: int = Field(description="Nombre de mots dans le chunk")


# ==============================================================================
# 2. NETTOYEUR DE TEXTE SCIENTIFIQUE
# ==============================================================================


def clean_scientific_text(text: str) -> str:
    """
    Nettoie le texte scientifique avant découpage :
    - Supprime les césures de mots de fin de ligne (ex: 'trans-\nformer' -> 'transformer')
    - Remplace les guillemets et apostrophes exotiques par des standards
    - Réduit les espaces multiples et les tabulations
    - Conserve le contenu mathématique lisible sans balises LaTeX lourdes
    """
    if not text:
        return ""

    # 1. Réunir les mots coupés par un tiret en fin de ligne
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # 2. Remplacer les retours à la ligne internes par des espaces simples
    text = re.sub(r"\r\n|\r|\n", " ", text)

    # 3. Normaliser les guillemets typographiques
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # 4. Supprimer les formules LaTeX trop complexes ou balises résiduelles
    text = re.sub(r"\$+", "", text)  # Retire les dollars mathématiques superflus
    text = re.sub(
        r"\\[a-zA-Z]+", " ", text
    )  # Retire les commandes LaTeX comme \textbf, \cite

    # 5. Supprimer les espaces multiples consécutifs
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==============================================================================
# 3. CLASSE PRINCIPALE : TextChunker
# ==============================================================================


class TextChunker:
    """
    Gère le découpage récursif et intelligent des articles en passages optimaux.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Initialise le splitter avec une stratégie récursive respectueuse de la ponctuation.

        L'ordre des séparateurs est crucial :
        1. On essaie d'abord de couper aux paragraphes ("\n\n")
        2. Sinon aux phrases (". ")
        3. Sinon aux points-virgules ou virgules ("; ", ", ")
        4. En dernier recours seulement aux espaces (" ")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            length_function=len,
        )
        logger.info(
            f"TextChunker configuré : taille={chunk_size} car., overlap={chunk_overlap} car."
        )

    def chunk_paper(self, paper: PaperMetadata) -> List[PaperChunk]:
        """
        Découpe un article donné en un ou plusieurs PaperChunks enrichis.

        Args:
            paper: L'objet PaperMetadata contenant le résumé et les métadonnées.

        Returns:
            Une liste d'objets PaperChunk prêts pour l'indexation.
        """
        # Nettoyage préalable de l'abstract
        cleaned_abstract = clean_scientific_text(paper.abstract)

        # Si l'abstract est très court (< chunk_size), on évite de le découper
        raw_chunks = self.splitter.split_text(cleaned_abstract)
        if not raw_chunks:
            return []

        chunks: List[PaperChunk] = []
        total = len(raw_chunks)

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Contextualisation : pour les embeddings, associer le titre au passage
            # permet au modèle de savoir de quel sujet traite l'extrait.
            search_content = f"Title: {paper.title}. Content: {chunk_text}"

            chunk_obj = PaperChunk(
                chunk_id=f"{paper.paper_id}_c{idx}",
                paper_id=paper.paper_id,
                chunk_index=idx,
                total_chunks=total,
                content=chunk_text,
                search_content=search_content,
                title=paper.title,
                authors=paper.authors,
                published_date=paper.published_date,
                primary_category=paper.primary_category,
                doi=paper.doi,
                entry_id=paper.entry_id,
                pdf_url=paper.pdf_url,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
            )
            chunks.append(chunk_obj)

        return chunks

    def process_metadata_file(
        self,
        input_file: Path = DEFAULT_METADATA_FILE,
        output_file: Path = DEFAULT_CHUNKS_FILE,
    ) -> List[PaperChunk]:
        """
        Lit l'ensemble des articles d'un fichier .jsonl, les découpe tous
        et enregistre les chunks résultants dans un nouveau fichier .jsonl.

        Args:
            input_file: Fichier source des métadonnées brutes.
            output_file: Fichier destination des chunks.

        Returns:
            La liste de tous les chunks générés.
        """
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Le fichier source {input_path} est introuvable. Veuillez d'abord collecter des données."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Lecture des articles depuis : {input_path}")
        papers: List[PaperMetadata] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    papers.append(PaperMetadata(**json.loads(line)))

        logger.info(f"{len(papers)} articles chargés. Découpage en cours...")

        all_chunks: List[PaperChunk] = []
        with open(output_path, "w", encoding="utf-8") as out:
            for paper in papers:
                chunks = self.chunk_paper(paper)
                for c in chunks:
                    out.write(c.model_dump_json() + "\n")
                    all_chunks.append(c)

        logger.info(
            f"Succès ! {len(all_chunks)} chunks générés à partir de {len(papers)} articles."
        )
        logger.info(f"Fichier de sortie : {output_path}")

        return all_chunks
