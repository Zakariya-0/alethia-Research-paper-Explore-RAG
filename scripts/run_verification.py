"""
Script de Démonstration et de Certification de la Vérification Logique de Claims Scientifiques.

Pipeline complet (Phase 3.1 à 3.4) :
1. Recherche Hybride (FAISS Dense + BM25 Sparse via Reciprocal Rank Fusion)
2. Reclassement sémantique par Cross-Encoder (ms-marco-MiniLM-L-6-v2)
3. Inférence logique NLI avec DeBERTa-v3 sur chaque passage
4. Vérification textuelle déterministe mot pour mot des citations (Anti-Hallucination)
5. Calcul du Score de Confiance Épistémique Composite (Consensus, Reranker, NLI, Récence)
6. Génération du Rapport Structuré standardisé Pydantic (ClaimVerificationReport)

Exemples d'utilisation :
-------------------------
1. Vérification d'une affirmation scientifique :
   python scripts/run_verification.py --claim "Video world models retain visual memory of generated content"

2. Export au format JSON structuré Pydantic :
   python scripts/run_verification.py --claim "Diffusion models improve image synthesis" --json
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

from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.verification.nli_evaluator import NLIEvaluator, StanceResult
from src.verification.citation_verifier import CitationVerifier
from src.verification.confidence_scorer import ConfidenceScorer
from src.verification.schemas import EvidenceItem, ClaimVerificationReport


def main():
    parser = argparse.ArgumentParser(
        description="Vérification formelle de claims scientifiques par RAG Hybride, DeBERTa NLI et Confiance Composite."
    )

    parser.add_argument(
        "--claim",
        type=str,
        default="Video world models retain visual memory of generated content",
        help="L'affirmation scientifique à vérifier formellement",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de passages de preuves à évaluer (par défaut: 3)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Afficher le rapport complet sous forme de JSON structuré",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Chemin de fichier optionnel pour enregistrer le rapport JSON",
    )

    args = parser.parse_args()

    if not args.json:
        print("\n" + "=" * 80)
        print("🔬 PLATEFORME DE VÉRIFICATION SCIENTIFIQUE — PIPELINE COMPLET PHASE 3")
        print("=" * 80)
        print(f'• Affirmation (Claim) : "{args.claim}"')
        print(f"• Passages analysés   : Top-{args.top_k}")
        print("=" * 80 + "\n")

    # [1/4] Initialisation des moteurs de recherche et reranking
    hybrid = HybridRetriever()
    reranker = CrossEncoderReranker()

    # [2/4] Chargement de DeBERTa-v3 et du vérificateur déterministe
    nli = NLIEvaluator()
    citation_verifier = CitationVerifier()
    scorer = ConfidenceScorer()

    # Étape 1 : Recherche en 2 temps (Hybride RRF + Reranker Cross-Encoder)
    if not args.json:
        print("\n[1/3] Recherche hybride et reclassement Cross-Encoder...")
    ranked_passages = hybrid.search_and_rerank(
        query=args.claim, top_k=args.top_k, num_candidates=12, reranker=reranker
    )

    if not ranked_passages:
        print(
            "[ERREUR] Aucun passage pertinent trouvé dans la base pour cette affirmation."
        )
        return

    # Étape 2 : Évaluation logique NLI par DeBERTa-v3
    if not args.json:
        print("[2/3] Évaluation de posture logique NLI (DeBERTa-v3)...")
    evaluated_passages = nli.evaluate_passages(
        claim=args.claim, passages=ranked_passages
    )

    # Étape 3 : Agrégation du Verdict Global et calcul de la confiance composite
    if not args.json:
        print("[3/3] Calcul du score composite et vérification des citations...")
    summary = nli.aggregate_verdict(evaluated_passages)
    confidence_breakdown = scorer.compute_confidence(
        evaluated_passages=evaluated_passages, global_verdict=summary["global_verdict"]
    )

    # Étape 4 : Construction des éléments de preuve certifiés (EvidenceItem)
    evidence_items = []
    for p in evaluated_passages:
        raw_text = p.get("raw_content", "")
        # Vérification déterministe sur un extrait textuel significatif du passage
        quote_sample = raw_text[: min(120, len(raw_text))].strip()
        audit_result = citation_verifier.verify_citation(
            citation=quote_sample, source_text=raw_text
        )

        stance_dict = p["nli_stance"]
        stance_obj = StanceResult(
            label=stance_dict["label"],
            confidence=stance_dict["confidence"],
            probabilities=stance_dict["probabilities"],
            is_conclusive=stance_dict["is_conclusive"],
            verdict_fr=stance_dict["verdict_fr"],
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
            citation_check=audit_result,
        )
        evidence_items.append(item)

    # Étape 5 : Construction du rapport officiel Pydantic
    report = ClaimVerificationReport(
        claim=args.claim,
        verdict=summary["global_verdict"],
        verdict_fr=summary["verdict_fr"],
        confidence_score=confidence_breakdown.composite_score,
        confidence_breakdown=confidence_breakdown,
        supports_count=summary["supports_count"],
        refutes_count=summary["refutes_count"],
        neutral_count=summary["neutral_count"],
        total_evidence_evaluated=summary["total_evaluated"],
        evidences=evidence_items,
    )

    # Mode Sortie JSON (si demandé par flag)
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        # Affichage synthétique console
        print("\n" + "=" * 80)
        print(f"⚖️ VERDICT GLOBAL : {report.verdict_fr}")
        print(f"• Score de certitude composite  : {report.confidence_score * 100:.1f}%")
        print(
            f"• Répartition des preuves        : {report.supports_count} Soutient | {report.refutes_count} Contredit | {report.neutral_count} Neutre"
        )
        print("=" * 80)

        # Décomposition du score de confiance
        cb = report.confidence_breakdown
        print("\n📊 DÉCOMPOSITION DU SCORE DE CONFIANCE COMPOSITE :")
        print("-" * 80)
        print(
            f"1. Consensus scientifique  (Poids {cb.weights['consensus']*100:.0f}%) : {cb.score_consensus * 100:.1f}%"
        )
        print(
            f"2. Pertinence Cross-Encoder (Poids {cb.weights['reranker']*100:.0f}%) : {cb.score_reranker * 100:.1f}%"
        )
        print(
            f"3. Déduction logique NLI   (Poids {cb.weights['nli']*100:.0f}%) : {cb.score_nli * 100:.1f}%"
        )
        print(
            f"4. Récence temporelle      (Poids {cb.weights['recency']*100:.0f}%) : {cb.score_recency * 100:.1f}%"
        )
        print(
            f"➜ Formule pondérée : S = 0.35*Consensus + 0.25*Rerank + 0.25*NLI + 0.15*Récence"
        )
        print(f"➜ Justification    : {cb.explanation}")
        print("-" * 80)

        # Détail des preuves individuelles
        print("\n📑 DÉTAIL DES ARTICLES & ANALYSE LOGIQUE PAR DEBERTA :")
        print("=" * 80)

        for i, ev in enumerate(report.evidences, start=1):
            stance = ev.nli_stance
            label = stance.label

            if label == "SUPPORTS":
                badge = "🟢 SOUTIENT"
            elif label == "REFUTES":
                badge = "🔴 CONTREDIT"
            else:
                badge = "⚪ NEUTRE"

            print(
                f"\n--- PREUVE #{i} : {badge} (Confiance NLI : {stance.confidence * 100:.1f}%) ---"
            )
            print(f"• Titre de l'article : {ev.title}")
            print(
                f"• Papier ID          : {ev.paper_id} ({ev.published_date}) | Catégorie: {ev.primary_category}"
            )
            print(f"• Lien arXiv         : {ev.entry_id}")
            print(
                f"• Pertinence Rerank  : {ev.rerank_score * 100:.1f}% (Rang initial: #{ev.original_hybrid_rank} ➜ #{ev.final_rank})"
            )
            print(
                f"• Distribution NLI   : Soutient={stance.probabilities['SUPPORTS']:.1%} | Contredit={stance.probabilities['REFUTES']:.1%} | Neutre={stance.probabilities['NEUTRAL']:.1%}"
            )
            print(f"• Extrait source analysé :")
            print(f'  "{ev.raw_content[:300]}..."')

            if ev.citation_check:
                status_txt = (
                    "✅ VALIDÉ MOT POUR MOT (EXACT MATCH)"
                    if ev.citation_check.is_valid
                    else "❌ NON CONCORDANT"
                )
                print(f"• Audit anti-hallucination : {status_txt}")

        print("\n" + "=" * 80 + "\n")

    # Sauvegarde sur disque si demandé
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        print(
            f"[OK] Rapport complet enregistré avec succès dans : {out_path.resolve()}"
        )


if __name__ == "__main__":
    main()
