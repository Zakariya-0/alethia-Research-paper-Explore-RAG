"""
Module des Nœuds Agents pour le Pipeline LangGraph Multi-Agents.

Rôle dans le projet :
----------------------
Définit les 5 nœuds d'agents spécialisés qui s'exécutent sur le graphe d'états :
1. `analyzer_node` (Agent Analyseur) : Analyse lexicale, mots-clés, catégories arXiv et reformulation.
2. `retriever_node` (Agent Récupérateur) : Pilotage de la recherche hybride (FAISS + BM25) et Reranker.
3. `evaluator_node` (Agent Évaluateur) : Inférence logique DeBERTa-v3 et détection du besoin de rétroaction.
4. `verifier_node` (Agent Vérificateur) : Audit textuel déterministe mot pour mot des citations.
5. `synthesis_node` (Agent Synthèse) : Calcul du score de confiance composite et rapport final Pydantic.

Optimisation CPU :
Les modèles lourds (DeBERTa, Cross-Encoder, Embeddings) sont mis en cache via un
gestionnaire singleton afin d'éviter tout rechargement superflu lors des rétroactions.
"""

import json
import re
import sys
import logging
from typing import Dict, Any, List, Optional

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import (
    DEFAULT_ARXIV_CATEGORIES,
    DEFAULT_METADATA_FILE,
    RERANK_MIN_RELEVANCE,
)
from src.data.chunker import clean_scientific_text
from src.generation import generate_answer, llm_status, rewrite_query
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.verification.nli_evaluator import NLIEvaluator, StanceResult
from src.verification.citation_verifier import CitationCheckResult, CitationVerifier
from src.verification.confidence_scorer import ConfidenceScorer
from src.verification.schemas import (
    AnswerCitation,
    ClaimVerificationReport,
    EvidenceItem,
    GeneratedAnswer,
    LLMInfo,
)
from src.agents.state import VerificationState, format_agent_log

logger = logging.getLogger(__name__)


# ==============================================================================
# GESTIONNAIRE SINGLETON POUR LES MODÈLES IA (OPTIMISATION MÉMOIRE CPU)
# ==============================================================================


class ModelRegistry:
    """Gestionnaire centralisé pour éviter de réinstancier les modèles sur CPU."""

    _hybrid: Optional[HybridRetriever] = None
    _reranker: Optional[CrossEncoderReranker] = None
    _nli: Optional[NLIEvaluator] = None
    _verifier: Optional[CitationVerifier] = None
    _scorer: Optional[ConfidenceScorer] = None
    _abstracts: Optional[Dict[str, str]] = None

    @classmethod
    def get_abstracts(cls) -> Dict[str, str]:
        """Abstracts arXiv stockés à la collecte (paper_id -> texte) : source de l'audit des citations."""
        if cls._abstracts is None:
            cls._abstracts = {}
            if DEFAULT_METADATA_FILE.exists():
                with open(DEFAULT_METADATA_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            paper = json.loads(line)
                            cls._abstracts[paper["paper_id"]] = paper["abstract"]
        return cls._abstracts

    @classmethod
    def get_hybrid(cls) -> HybridRetriever:
        if cls._hybrid is None:
            cls._hybrid = HybridRetriever()
        return cls._hybrid

    @classmethod
    def get_reranker(cls) -> CrossEncoderReranker:
        if cls._reranker is None:
            cls._reranker = CrossEncoderReranker()
        return cls._reranker

    @classmethod
    def get_nli(cls) -> NLIEvaluator:
        if cls._nli is None:
            cls._nli = NLIEvaluator()
        return cls._nli

    @classmethod
    def get_verifier(cls) -> CitationVerifier:
        if cls._verifier is None:
            cls._verifier = CitationVerifier()
        return cls._verifier

    @classmethod
    def get_scorer(cls) -> ConfidenceScorer:
        if cls._scorer is None:
            cls._scorer = ConfidenceScorer()
        return cls._scorer


# ==============================================================================
# 1. AGENT ANALYSEUR (Query Analysis & Decomposer)
# ==============================================================================

# Stopwords scientifiques génériques pour l'extraction de mots-clés
SCIENTIFIC_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "up",
    "about",
    "into",
    "over",
    "after",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "must",
    "that",
    "which",
    "who",
    "whom",
    "this",
    "these",
    "those",
    "it",
    "its",
    "they",
    "their",
    "we",
    "our",
    "you",
    "your",
    "uses",
    "used",
    "using",
    "use",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "papers",
    "approach",
}

# Règles heuristiques de détection de catégories arXiv
CATEGORY_RULES = {
    "cs.CV": [
        "vision",
        "visual",
        "image",
        "video",
        "segmentation",
        "depth",
        "camera",
        "frustum",
        "render",
        "diffusion",
    ],
    "cs.CL": [
        "language",
        "nlp",
        "translation",
        "summarization",
        "text",
        "vlm",
        "linguistic",
        "sentence",
        "semantic",
    ],
    "cs.LG": [
        "learning",
        "reinforcement",
        "neural",
        "deep",
        "gradient",
        "loss",
        "weights",
        "transformer",
        "generalization",
    ],
    "cs.AI": [
        "intelligence",
        "agent",
        "agents",
        "reasoning",
        "planning",
        "logic",
        "knowledge",
        "heuristic",
    ],
    "cs.SE": [
        "software",
        "code",
        "bug",
        "debugging",
        "repository",
        "git",
        "programming",
        "developer",
        "refactoring",
    ],
}


def extract_scientific_keywords(text: str) -> List[str]:
    """Extrait les termes techniques saillants en ignorant les mots vides."""
    tokens = re.findall(r"[A-Za-z0-9_-]{2,}", text)
    keywords = [t for t in tokens if t.lower() not in SCIENTIFIC_STOPWORDS]
    # Déduplication tout en préservant l'ordre
    seen = set()
    result = []
    for k in keywords:
        if k.lower() not in seen:
            seen.add(k.lower())
            result.append(k)
    return result


def detect_target_categories(keywords: List[str]) -> List[str]:
    """Associe les mots-clés extraits aux catégories arXiv pertinentes."""
    matched_cats = set()
    lowered_kw = [k.lower() for k in keywords]

    for cat, patterns in CATEGORY_RULES.items():
        for pat in patterns:
            if any(pat in kw for kw in lowered_kw):
                matched_cats.add(cat)
                break

    if not matched_cats:
        return DEFAULT_ARXIV_CATEGORIES[:2]  # Fallback : cs.AI et cs.CL
    return sorted(list(matched_cats))


def analyzer_node(state: VerificationState) -> Dict[str, Any]:
    """
    Nœud 1 : Analyse l'affirmation, extrait les concepts scientifiques,
    sélectionne les catégories cibles, ou reformule la requête en cas de feedback.
    """
    iteration = state.get("iteration_count", 0)
    claim = state.get("claim", "")

    # CAS A : Première passe d'analyse (Round 0)
    if iteration == 0:
        keywords = extract_scientific_keywords(claim)
        target_cats = detect_target_categories(keywords)
        current_query = claim

        log = format_agent_log(
            agent_name="Agent Analyseur",
            action="ANALYSE_INITIALE",
            details=f"Concepts extraits: {keywords[:5]} | Catégories arXiv ciblées: {target_cats}",
        )

        return {
            "current_query": current_query,
            "extracted_keywords": keywords,
            "target_categories": target_cats,
            "query_reformulation_notes": "Formulation initiale directe",
            "query_history": [current_query],
            "agent_logs": [log],
        }

    # CAS B : Boucle de rétroaction active (Round > 0)
    keywords = state.get("extracted_keywords", [])
    if not keywords:
        keywords = extract_scientific_keywords(claim)

    # Stratégie de reformulation : le LLM propose une requête différente des
    # précédentes ; sans LLM, repli sur la focalisation par mots-clés
    # (rappel lexical BM25 et alignement sémantique FAISS)
    history = list(state.get("query_history", []))
    key_terms = " ".join(keywords[:6])
    llm_query = (
        rewrite_query(claim, history) if llm_status()["enabled"] else None
    )
    reformulated_query = llm_query or f"{claim} {key_terms}"

    # Élargir les catégories arXiv au pool général pour maximiser la découverte
    expanded_cats = sorted(
        list(set(state.get("target_categories", []) + ["cs.AI", "cs.LG"]))
    )

    if llm_query:
        notes = f"Reformulation LLM (Round {iteration}) : requête [{llm_query}] avec élargissement aux catégories {expanded_cats}"
    else:
        notes = f"Reformulation active (Round {iteration}) : focalisation sur [{key_terms}] avec élargissement aux catégories {expanded_cats}"

    log = format_agent_log(
        agent_name="Agent Analyseur", action="REFORMULATION_FEEDBACK", details=notes
    )

    return {
        "current_query": reformulated_query,
        "target_categories": expanded_cats,
        "query_reformulation_notes": notes,
        "query_history": history + [reformulated_query],
        "agent_logs": [log],
    }


# ==============================================================================
# 2. AGENT RÉCUPÉRATEUR (Retriever & Reranker Controller)
# ==============================================================================


def retriever_node(state: VerificationState) -> Dict[str, Any]:
    """
    Nœud 2 : Pilote le moteur RAG en 2 temps :
    1. Recherche hybride (FAISS + BM25 via RRF)
    2. Reclassement par Cross-Encoder
    """
    query = state.get("current_query", state.get("claim", ""))
    target_cats = state.get("target_categories", [])
    primary_filter = target_cats[0] if len(target_cats) == 1 else None

    hybrid = ModelRegistry.get_hybrid()
    reranker = ModelRegistry.get_reranker()

    # Exécution de la recherche hybride avec reclassement Cross-Encoder
    ranked = hybrid.search_and_rerank(
        query=query,
        top_k=3,
        num_candidates=12,
        reranker=reranker,
        category_filter=primary_filter,
    )

    # La requête a pu être reformulée (par le LLM) : la pertinence se juge toujours
    # contre le claim d'origine, sinon une reformulation qui dérive de sujet
    # ferait passer des passages hors sujet.
    claim = state.get("claim", query)
    if ranked and query != claim:
        ranked = reranker.rerank(query=claim, documents=ranked, top_k=len(ranked))

    # Garde-fou de pertinence : le top-k est toujours renvoyé, même hors sujet.
    # Les passages sous le seuil ne partent donc pas vers l'évaluation NLI.
    passages = [p for p in ranked if p.get("rerank_score", 0.0) >= RERANK_MIN_RELEVANCE]
    discarded = len(ranked) - len(passages)

    if not ranked:
        retrieval_status = "EMPTY"
        status_msg = "Aucun passage trouvé dans le corpus"
    elif not passages:
        retrieval_status = "LOW_RELEVANCE"
        status_msg = (
            f"{len(ranked)} passages trouvés mais tous écartés : "
            f"pertinence sémantique < {RERANK_MIN_RELEVANCE * 100:.0f}%"
        )
    else:
        avg_score = sum(p.get("rerank_score", 0.0) for p in passages) / len(passages)
        if avg_score < 0.25:
            retrieval_status = "LOW_RELEVANCE"
            status_msg = f"{len(passages)} passages trouvés mais pertinence sémantique faible ({avg_score*100:.1f}%)"
        else:
            retrieval_status = "SUCCESS"
            status_msg = f"{len(passages)} passages pertinents extraits et reclassés (score moyen: {avg_score*100:.1f}%)"
        if discarded:
            status_msg += f" | {discarded} passage(s) écarté(s) sous le seuil de pertinence"

    log = format_agent_log(
        agent_name="Agent Récupérateur",
        action="RECUPERATION_HYBRIDE",
        details=status_msg,
    )

    return {
        "retrieved_passages": passages,
        "retrieval_status": retrieval_status,
        "discarded_passages": discarded,
        "agent_logs": [log],
    }


# ==============================================================================
# 3. AGENT ÉVALUATEUR (Stance & Logic Evaluator)
# ==============================================================================


def evaluator_node(state: VerificationState) -> Dict[str, Any]:
    """
    Nœud 3 : Évalue la posture logique (DeBERTa-v3 NLI) de chaque passage retenu,
    agrège le consensus et décide s'il faut déclencher la boucle de rétroaction.
    """
    claim = state.get("claim", "")
    passages = state.get("retrieved_passages", [])
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_feedback_iterations", 2)

    nli = ModelRegistry.get_nli()

    # Si aucun passage n'a été récupéré
    if not passages:
        verdict_summary = {
            "global_verdict": "PREUVES_INSUFFISANTES",
            "verdict_fr": "Aucune source scientifique trouvée",
            "confidence": 0.0,
            "supports_count": 0,
            "refutes_count": 0,
            "neutral_count": 0,
            "total_evaluated": 0,
        }
        evaluated_passages = []
    else:
        evaluated_passages = nli.evaluate_passages(claim=claim, passages=passages)
        verdict_summary = nli.aggregate_verdict(evaluated_passages)

    # Logique d'Arbitrage pour la Boucle de Rétroaction Active (Active Feedback Loop)
    # Si le verdict est PREUVES_INSUFFISANTES et qu'il reste des itérations autorisées
    is_insufficient = verdict_summary["global_verdict"] == "PREUVES_INSUFFISANTES"
    has_retries_left = iteration < max_iter

    if is_insufficient and has_retries_left:
        needs_feedback = True
        next_iter = iteration + 1
        reason = f"Preuves insuffisantes au tour {next_iter}/{max_iter}. Déclenchement de la rétroaction pour affiner la recherche."
        feedback_entry = f"[Tour {next_iter}] Échec de consensus concluant -> Rétroaction vers l'Agent Analyseur"
    else:
        needs_feedback = False
        next_iter = iteration
        reason = None
        feedback_entry = None

    log_details = (
        f"Verdict provisoire: {verdict_summary['global_verdict']} "
        f"({verdict_summary['supports_count']}S / {verdict_summary['refutes_count']}R / {verdict_summary['neutral_count']}N) | "
        f"Rétroaction requise: {'OUI' if needs_feedback else 'NON'}"
    )

    log = format_agent_log(
        agent_name="Agent Évaluateur",
        action="EVALUATION_LOGIQUE_NLI",
        details=log_details,
    )

    update = {
        "evaluated_passages": evaluated_passages,
        "verdict_summary": verdict_summary,
        "needs_feedback": needs_feedback,
        "feedback_reason": reason,
        "iteration_count": next_iter,
        "agent_logs": [log],
    }

    if feedback_entry:
        update["feedback_history"] = [feedback_entry]

    return update


# ==============================================================================
# 3 bis. AGENT GÉNÉRATEUR (Réponse rédigée par le LLM, fondée sur les passages)
# ==============================================================================


def generator_node(state: VerificationState) -> Dict[str, Any]:
    """
    Nœud 3 bis : fait rédiger par le LLM (init_chat_model) une réponse fondée
    uniquement sur les passages retenus, avec citations. Le nœud ne lève jamais
    d'erreur : sans LLM, sans passage ou en cas d'échec, le pipeline continue
    avec son verdict NLI et le statut est reporté dans `llm_info`.
    """
    claim = state.get("claim", "")
    passages = state.get("evaluated_passages", [])
    status = llm_status()

    info: Dict[str, Any] = {
        "enabled": status["enabled"],
        "provider": status["provider"],
        "model": status["model"],
        "status": "OK",
        "message": None,
        "latency_s": None,
    }
    answer: Optional[Dict[str, Any]] = None

    if not status["enabled"]:
        info.update(status="DISABLED", message=status["reason"])
    elif not passages:
        info.update(
            status="SKIPPED",
            message="Aucun passage pertinent : aucune réponse rédigée",
        )
    else:
        result = generate_answer(claim, passages)
        info["latency_s"] = result["latency_s"]
        if result["ok"]:
            answer = result["answer"].model_dump()
        else:
            info.update(status="ERROR", message=result["error"])

    if info["status"] == "OK":
        details = (
            f"Réponse rédigée par {info['provider']}:{info['model']} en "
            f"{info['latency_s']} s | {len(answer['citations'])} citation(s) à auditer"
        )
    else:
        details = f"{info['status']} : {info['message']}"

    log = format_agent_log(
        agent_name="Agent Générateur", action="GENERATION_LLM", details=details
    )
    return {"llm_answer": answer, "llm_info": info, "agent_logs": [log]}


# ==============================================================================
# 4. AGENT VÉRIFICATEUR (Citation & Fact Auditor)
# ==============================================================================


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")


def select_evidence_sentence(text: str, claim: str) -> str:
    """
    Choisit dans un passage la phrase la plus proche du claim (recouvrement
    lexical, déterministe) : c'est cette phrase qui est citée puis auditée.
    """
    sentences = [s.strip().lstrip(". ") for s in _SENTENCE_SPLIT.split(text)]
    sentences = [s for s in sentences if len(s.split()) >= 3]
    if not sentences:
        return text.strip()

    claim_terms = set(_WORD.findall(" ".join(extract_scientific_keywords(claim)).lower()))

    def overlap(sentence: str) -> int:
        return len(claim_terms & set(_WORD.findall(sentence.lower())))

    # max() renvoie la première phrase en cas d'égalité
    return max(sentences, key=overlap)


def audit_citation(
    verifier: CitationVerifier, quote: str, paper_id: str
) -> CitationCheckResult:
    """
    Vérifie que la citation figure à l'identique (casse comprise) dans l'abstract
    de l'article tel que stocké à la collecte, brut puis nettoyé comme au chunking.
    Aucun appariement approximatif n'est accepté.
    """
    abstract = ModelRegistry.get_abstracts().get(paper_id, "")
    result = CitationCheckResult(
        is_valid=False, match_type="NOT_FOUND", similarity_ratio=0.0
    )
    if not abstract:
        return result

    normalized_quote = verifier.normalize_for_check(quote)
    for source in (abstract, clean_scientific_text(abstract)):
        # min_fuzzy_ratio=1.0 : pas de correspondance floue
        result = verifier.verify_citation(
            citation=quote, source_text=source, min_fuzzy_ratio=1.0
        )
        # verify_citation ignore la casse : contrôle strict en plus
        if result.is_valid and normalized_quote in verifier.normalize_for_check(source):
            return result
    return result.model_copy(update={"is_valid": False, "match_type": "NOT_FOUND"})


def verifier_node(state: VerificationState) -> Dict[str, Any]:
    """
    Nœud 4 : Audit textuel déterministe mot pour mot (Anti-Hallucination).
    Cite la phrase du passage la plus proche du claim et contrôle qu'elle figure
    à l'identique dans l'abstract arXiv stocké à la collecte (source distincte du chunk).
    """
    claim = state.get("claim", "")
    evaluated = state.get("evaluated_passages", [])
    verifier = ModelRegistry.get_verifier()

    evidence_items: List[Dict[str, Any]] = []
    failed_audits = 0

    for p in evaluated:
        raw_text = p.get("raw_content", "")
        quote = select_evidence_sentence(raw_text, claim)
        audit = audit_citation(verifier, quote, p.get("paper_id", ""))

        if not audit.is_valid:
            failed_audits += 1

        stance_data = p.get("nli_stance", {})
        stance_obj = StanceResult(
            label=stance_data.get("label", "NEUTRAL"),
            confidence=stance_data.get("confidence", 0.0),
            probabilities=stance_data.get("probabilities", {}),
            is_conclusive=stance_data.get("is_conclusive", False),
            verdict_fr=stance_data.get("verdict_fr", ""),
        )

        item = EvidenceItem(
            paper_id=p.get("paper_id", "inconnu"),
            chunk_id=p.get("chunk_id", ""),
            title=p.get("title", "Sans titre"),
            authors=p.get("authors", []),
            published_date=p.get("published_date", "2024-01-01"),
            primary_category=p.get("primary_category", "cs.AI"),
            entry_id=p.get("entry_id", ""),
            pdf_url=p.get("pdf_url"),
            doi=p.get("doi"),
            raw_content=raw_text,
            rerank_score=float(p.get("rerank_score", 0.0)),
            original_hybrid_rank=p.get("original_hybrid_rank"),
            final_rank=p.get("final_rank"),
            nli_stance=stance_obj,
            evidence_quote=quote,
            citation_check=audit,
        )
        evidence_items.append(item.model_dump())

    audits_passed = failed_audits == 0

    # Audit des citations du LLM : chaque phrase citée doit figurer à l'identique
    # dans l'abstract de l'article désigné, sinon elle est marquée non vérifiée.
    llm_answer = state.get("llm_answer")
    llm_note = ""
    if llm_answer:
        passages_by_chunk = {p.get("chunk_id"): p for p in evaluated}
        audited: List[Dict[str, Any]] = []
        for cite in llm_answer.get("citations", []):
            source = passages_by_chunk.get(cite["chunk_id"])
            if source is None:
                # chunk_id inventé ou hors des passages fournis au LLM
                check = CitationCheckResult(
                    is_valid=False, match_type="NOT_FOUND", similarity_ratio=0.0
                )
            else:
                check = audit_citation(verifier, cite["quote"], source["paper_id"])
            audited.append(
                {
                    "chunk_id": cite["chunk_id"],
                    "paper_id": source["paper_id"] if source else "",
                    "title": source["title"] if source else "",
                    "quote": cite["quote"],
                    "is_verified": check.is_valid,
                    "citation_check": check.model_dump(),
                }
            )
        llm_answer = {**llm_answer, "citations": audited}
        rejected = sum(1 for c in audited if not c["is_verified"])
        llm_note = f" | Citations du LLM : {len(audited) - rejected}/{len(audited)} vérifiées"

    log = format_agent_log(
        agent_name="Agent Vérificateur",
        action="AUDIT_CITATIONS",
        details=(
            f"{len(evidence_items)} citations auditées dans l'abstract stocké | "
            + (
                "Intégrité textuelle: 100% VALIDE"
                if audits_passed
                else f"ANOMALIE DÉTECTÉE ({failed_audits} citation(s) non retrouvée(s))"
            )
            + llm_note
        ),
    )

    return {
        "evidences": evidence_items,
        "citation_audit_passed": audits_passed,
        "llm_answer": llm_answer,
        "agent_logs": [log],
    }


# ==============================================================================
# 5. AGENT SYNTHÈSE (Synthesis & Report Generator)
# ==============================================================================


# Correspondance entre le verdict issu de DeBERTa et la position rendue par le LLM
_STANCE_BY_VERDICT = {
    "SOUTENU": "SUPPORTS",
    "REFUTE": "REFUTES",
    "SUJET_DEBATTU": "MIXED",
    "PREUVES_INSUFFISANTES": "INSUFFICIENT",
}


def synthesis_node(state: VerificationState) -> Dict[str, Any]:
    """
    Nœud 5 : Calcule le score de confiance composite, compile les preuves
    et génère le rapport officiel consolidé (ClaimVerificationReport).
    """
    claim = state.get("claim", "")
    evaluated = state.get("evaluated_passages", [])
    summary = state.get("verdict_summary", {})
    evidences_raw = state.get("evidences", [])

    scorer = ModelRegistry.get_scorer()
    breakdown = scorer.compute_confidence(
        evaluated_passages=evaluated,
        global_verdict=summary.get("global_verdict", "PREUVES_INSUFFISANTES"),
    )

    # Reconstruction des objets EvidenceItem
    evidence_objs = [EvidenceItem(**e) for e in evidences_raw]

    # Réponse rédigée par le LLM (None si LLM inactif, ignoré ou en erreur)
    verdict = summary.get("global_verdict", "PREUVES_INSUFFISANTES")
    llm_answer = state.get("llm_answer")
    answer_obj = None
    if llm_answer:
        citations = [AnswerCitation(**c) for c in llm_answer.get("citations", [])]
        answer_obj = GeneratedAnswer(
            text=llm_answer["answer"],
            stance=llm_answer["stance"],
            citations=citations,
            # sans citation, seule une réponse "INSUFFICIENT" est légitime
            citations_verified=(
                all(c.is_verified for c in citations)
                if citations
                else llm_answer["stance"] == "INSUFFICIENT"
            ),
            agrees_with_nli=_STANCE_BY_VERDICT.get(verdict) == llm_answer["stance"],
        )
    llm_info = state.get("llm_info")

    # Génération du rapport officiel
    report = ClaimVerificationReport(
        claim=claim,
        verdict=summary.get("global_verdict", "PREUVES_INSUFFISANTES"),
        verdict_fr=summary.get("verdict_fr", "Preuves insuffisantes"),
        confidence_score=breakdown.composite_score,
        confidence_breakdown=breakdown,
        supports_count=summary.get("supports_count", 0),
        refutes_count=summary.get("refutes_count", 0),
        neutral_count=summary.get("neutral_count", 0),
        total_evidence_evaluated=summary.get("total_evaluated", len(evaluated)),
        evidences=evidence_objs,
        answer=answer_obj,
        llm=LLMInfo(**llm_info) if llm_info else None,
    )

    log = format_agent_log(
        agent_name="Agent Synthèse",
        action="SYNTHESE_FINALE",
        details=f"Verdict: {report.verdict} | Confiance Composite: {report.confidence_score * 100:.1f}% | Rapport Pydantic certifié généré",
    )

    return {
        "confidence_breakdown": breakdown.model_dump(),
        "final_report": report.model_dump(),
        "agent_logs": [log],
    }
