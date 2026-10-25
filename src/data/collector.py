"""
Module de collecte des articles scientifiques depuis l'API arXiv.

Ce script permet d'extraire de manière automatisée :
- Les métadonnées complètes (Titre, Auteurs, Date, Catégories, DOI, Résumé).
- L'URL vers le document et le lien PDF direct.
- Optionnellement, le téléchargement du fichier PDF en local.

Le format de sauvegarde utilisé est le JSON Lines (.jsonl) :
Chaque ligne du fichier est un objet JSON autonome. Cela permet d'écrire
au fur et à mesure sans tout charger en mémoire et d'éviter toute perte
de données en cas d'interruption du script.
"""

import os
import sys
import json
import time
import logging
from typing import List, Optional, Set, Dict, Any
from pathlib import Path
from datetime import datetime

# Encodage sécurisé pour la console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import arxiv
from pydantic import BaseModel, Field

# Import de la configuration centralisée
from src.config import (
    RAW_DATA_DIR,
    PDFS_DIR,
    DEFAULT_METADATA_FILE,
    DEFAULT_ARXIV_CATEGORIES,
    ARXIV_DELAY_SECONDS,
    ARXIV_BATCH_SIZE,
)

# Configuration des logs pour un affichage propre dans la console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# MODÈLE DE DONNÉES : Schéma d'un article scientifique (Pydantic)
# ==============================================================================


class PaperMetadata(BaseModel):
    """
    Modèle de données validant la structure des métadonnées d'un article arXiv.
    Garantit que chaque article collecté possède des types stricts et vérifiés.
    """

    paper_id: str = Field(description="Identifiant unique arXiv (ex: '2401.12345')")
    title: str = Field(description="Titre complet de l'article")
    authors: List[str] = Field(
        default_factory=list, description="Liste des noms des auteurs"
    )
    abstract: str = Field(description="Résumé / Abstract officiel de l'article")
    categories: List[str] = Field(
        default_factory=list,
        description="Toutes les catégories attribuées (ex: ['cs.AI', 'cs.LG'])",
    )
    primary_category: str = Field(
        description="Catégorie principale de l'article (ex: 'cs.AI')"
    )
    published_date: str = Field(
        description="Date de publication au format ISO (AAAA-MM-JJ)"
    )
    updated_date: str = Field(description="Date de dernière mise à jour au format ISO")
    doi: Optional[str] = Field(
        default=None, description="DOI (Digital Object Identifier) si disponible"
    )
    entry_id: str = Field(description="URL de la page web arXiv de l'article")
    pdf_url: str = Field(description="Lien de téléchargement direct du PDF")
    local_pdf_path: Optional[str] = Field(
        default=None, description="Chemin local du fichier PDF s'il a été téléchargé"
    )
    collected_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Date et heure de collecte",
    )


# ==============================================================================
# CLASSE PRINCIPALE : Collecteur d'articles arXiv
# ==============================================================================


class ArxivCollector:
    """
    Collecteur d'articles scientifiques exploitant l'API officielle d'arXiv.
    Gère la pagination, le respect des quotas (rate limiting) et la déduplication.
    """

    def __init__(
        self,
        output_file: Path = DEFAULT_METADATA_FILE,
        pdf_dir: Path = PDFS_DIR,
        delay_seconds: float = ARXIV_DELAY_SECONDS,
        page_size: int = ARXIV_BATCH_SIZE,
    ):
        """
        Initialise le collecteur.

        Args:
            output_file: Chemin du fichier JSONL de sauvegarde des métadonnées.
            pdf_dir: Dossier local où stocker les fichiers PDF téléchargés.
            delay_seconds: Temps d'attente (en secondes) entre les requêtes (3s recommandées par arXiv).
            page_size: Nombre de résultats demandés par page de requête.
        """
        self.output_file = Path(output_file)
        self.pdf_dir = Path(pdf_dir)
        self.delay_seconds = delay_seconds
        self.page_size = page_size

        # Création du client arXiv avec gestion personnalisée du délai de politesse
        self.client = arxiv.Client(
            page_size=self.page_size, delay_seconds=self.delay_seconds, num_retries=3
        )

        # Assurer que les dossiers de destination existent
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Charger la liste des identifiants déjà collectés pour éviter les doublons
        self.existing_ids: Set[str] = self._load_existing_ids()
        logger.info(
            f"Collecteur initialisé. {len(self.existing_ids)} articles déjà présents dans {self.output_file.name}"
        )

    def _load_existing_ids(self) -> Set[str]:
        """
        Lit le fichier JSONL existant pour recenser les identifiants déjà sauvegardés.
        Permet de reprendre une collecte interrompue sans re-télécharger les mêmes articles.
        """
        ids = set()
        if not self.output_file.exists():
            return ids

        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        if "paper_id" in data:
                            ids.add(data["paper_id"])
        except Exception as e:
            logger.warning(
                f"Erreur lors de la lecture du fichier existant {self.output_file}: {e}"
            )

        return ids

    @staticmethod
    def build_query(
        categories: Optional[List[str]] = None, keywords: Optional[List[str]] = None
    ) -> str:
        """
        Construit la requête de recherche au format attendu par l'API arXiv.
        Exemple : 'cat:cs.AI OR cat:cs.CL' ou '(cat:cs.AI) AND (transformer OR llm)'
        """
        parts = []

        # Filtre sur les catégories arXiv (ex: cs.AI, cs.CL)
        if categories:
            cat_query = " OR ".join([f"cat:{cat.strip()}" for cat in categories])
            parts.append(f"({cat_query})")

        # Filtre sur des mots-clés optionnels
        if keywords:
            kw_query = " OR ".join([f'all:"{kw.strip()}"' for kw in keywords])
            parts.append(f"({kw_query})")

        if not parts:
            # Par défaut : toutes les catégories informatiques ciblées
            return " OR ".join([f"cat:{cat}" for cat in DEFAULT_ARXIV_CATEGORIES])

        return " AND ".join(parts)

    def _convert_arxiv_result(self, result: arxiv.Result) -> PaperMetadata:
        """
        Convertit un objet résultat brut de la librairie arxiv en modèle structuré PaperMetadata.
        Nettoie le titre et le résumé pour supprimer les retours à la ligne parasites.
        """
        # Nettoyage du titre (arXiv inclut parfois des sauts de ligne '\n' au milieu du titre)
        clean_title = " ".join(result.title.split())

        # Nettoyage de l'abstract
        clean_abstract = " ".join(result.summary.split())

        # Extraction propre de l'ID (sans la version 'v1' pour avoir un ID canonique)
        raw_id = result.get_short_id()
        canonical_id = raw_id.split("v")[0] if "v" in raw_id else raw_id

        return PaperMetadata(
            paper_id=canonical_id,
            title=clean_title,
            authors=[author.name for author in result.authors],
            abstract=clean_abstract,
            categories=result.categories,
            primary_category=result.primary_category,
            published_date=result.published.strftime("%Y-%m-%d"),
            updated_date=result.updated.strftime("%Y-%m-%d"),
            doi=result.doi,
            entry_id=result.entry_id,
            pdf_url=result.pdf_url,
        )

    def save_paper(self, paper: PaperMetadata):
        """
        Sauvegarde un article dans le fichier JSONL (ajout à la fin du fichier).
        """
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(paper.model_dump_json() + "\n")
        self.existing_ids.add(paper.paper_id)

    def download_pdf(self, paper: PaperMetadata) -> Optional[str]:
        """
        Télécharge le fichier PDF associé à un article et enregistre son chemin local.
        """
        pdf_filename = f"{paper.paper_id.replace('/', '_')}.pdf"
        target_path = self.pdf_dir / pdf_filename

        # Si le PDF existe déjà en local, ne pas le re-télécharger
        if target_path.exists():
            return str(target_path)

        try:
            logger.info(f"Téléchargement du PDF pour l'article {paper.paper_id}...")
            # On recrée une recherche ciblée sur cet ID pour utiliser la méthode de téléchargement native d'arxiv
            paper_obj = next(
                self.client.results(arxiv.Search(id_list=[paper.paper_id]))
            )
            paper_obj.download_pdf(dirpath=str(self.pdf_dir), filename=pdf_filename)
            time.sleep(self.delay_seconds)  # Respect du délai
            return str(target_path)
        except Exception as e:
            logger.error(f"Échec du téléchargement du PDF pour {paper.paper_id}: {e}")
            return None

    def collect_papers(
        self,
        categories: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        max_results: int = 50,
        download_pdfs: bool = False,
    ) -> List[PaperMetadata]:
        """
        Lance la collecte d'articles selon les critères définis.

        Args:
            categories: Liste de catégories arXiv (ex: ['cs.AI', 'cs.CL']).
            keywords: Liste de mots-clés facultatifs.
            max_results: Nombre maximal d'articles à collecter.
            download_pdfs: Si True, télécharge aussi les PDFs complets sur le disque.

        Returns:
            Liste des nouveaux articles collectés et enregistrés.
        """
        query_str = self.build_query(categories=categories, keywords=keywords)
        logger.info(f"Lancement de la recherche avec la requête : '{query_str}'")
        logger.info(
            f"Objectif : {max_results} articles (Téléchargement PDF : {'Oui' if download_pdfs else 'Non'})"
        )

        # On demande un pool suffisant pour compenser les doublons déjà présents
        search_limit = max_results + len(self.existing_ids)
        search = arxiv.Search(
            query=query_str,
            max_results=search_limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,  # Récupérer les plus récents en premier
        )

        collected: List[PaperMetadata] = []
        skipped_count = 0

        try:
            for result in self.client.results(search):
                paper = self._convert_arxiv_result(result)

                # Éviter les doublons
                if paper.paper_id in self.existing_ids:
                    skipped_count += 1
                    continue

                # Téléchargement optionnel du PDF
                if download_pdfs:
                    local_pdf = self.download_pdf(paper)
                    paper.local_pdf_path = local_pdf

                # Sauvegarde immédiate dans le JSONL
                self.save_paper(paper)
                collected.append(paper)

                logger.info(
                    f"[+{len(collected)}] {paper.paper_id} : {paper.title[:65]}... ({paper.published_date})"
                )

                if len(collected) >= max_results:
                    break

        except Exception as e:
            logger.error(f"Erreur pendant l'itération des résultats arXiv : {e}")

        logger.info(
            f"Collecte terminée ! {len(collected)} nouveaux articles ajoutés ({skipped_count} doublons ignorés)."
        )
        logger.info(f"Total actuel dans le corpus : {len(self.existing_ids)} articles.")
        return collected
