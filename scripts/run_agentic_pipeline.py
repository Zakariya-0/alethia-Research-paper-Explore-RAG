"""
Script de Démonstration du Pipeline Multi-Agents Autonome LangGraph.

Ce script exécute le graphe d'états multi-agents :
1. Agent Analyseur : Extraction de concepts et ciblage de catégories arXiv.
2. Agent Récupérateur : Recherche hybride FAISS + BM25 et reclassement Cross-Encoder.
3. Agent Évaluateur : Inférence logique DeBERTa-v3 et détection de consensus.
   * BOUCLE DE RÉTROACTION ACTIVE : Si preuves insuffisantes, réitération avec reformulation.
4. Agent Vérificateur : Audit déterministe anti-hallucination mot pour mot.
5. Agent Synthèse : Score de confiance composite et rapport certifié Pydantic.

Exemples d'utilisation :
-------------------------
1. Vérification d'une affirmation scientifique :
   python scripts/run_agentic_pipeline.py --claim "Video world models retain visual memory of generated content"

2. Test avec affichage du journal détaillé des agents :
   python scripts/run_agentic_pipeline.py --claim "Velox uses spatial memory for video world models" --verbose

3. Export JSON officiel :
   python scripts/run_agentic_pipeline.py --claim "Diffusion models improve image synthesis" --json
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

from src.agents import verify_claim_agentic


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Multi-Agents LangGraph pour la vérification souveraine de claims scientifiques."
    )

    parser.add_argument(
        "--claim",
        type=str,
        default="Video world models retain visual memory of generated content",
        help="L'affirmation scientifique à soumettre au collège d'agents",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Nombre maximal de tours de rétroaction autorisés (défaut: 2)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher le journal d'audit complet de chaque agent pas à pas",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Afficher le rapport final au format JSON structuré",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Fichier de sortie pour exporter le rapport JSON",
    )

    args = parser.parse_args()

    if not args.json:
        print("\n" + "=" * 85)
        print(
            "🤖 PLATEFORME MULTI-AGENTS LANGGRAPH — VÉRIFICATION SCIENTIFIQUE SOUVERAINE"
        )
        print("=" * 85)
        print(f'• Affirmation (Claim)     : "{args.claim}"')
        print(f"• Max boucles rétroaction : {args.max_retries}")
        print("=" * 85 + "\n")
        print("⏳ Lancement de l'orchestration multi-agents LangGraph...\n")

    # Exécution du graphe d'états complet
    final_state = verify_claim_agentic(
        claim=args.claim, max_feedback_iterations=args.max_retries
    )

    report_dict = final_state.get("final_report", {})

    if args.json:
        print(json.dumps(report_dict, indent=2, ensure_ascii=False))
        return

    # 1. Visualisation du journal d'audit des agents (Timeline d'exécution)
    print("📜 HISTORIQUE DES ÉTAPES AGENTS (TIMELINE LANGGRAPH) :")
    print("-" * 85)
    for log in final_state.get("agent_logs", []):
        ts = log.get("timestamp", "")
        agent = log.get("agent", "")
        action = log.get("action", "")
        details = log.get("details", "")
        print(f"[{ts}] [{agent:<18}] {action:<22} ➜ {details}")
    print("-" * 85)

    # 2. Détection de rétroaction active
    fb_history = final_state.get("feedback_history", [])
    if fb_history:
        print("\n🔄 BOUCLES DE RÉTROACTION ACTIVE DÉCLENCHÉES :")
        for fb in fb_history:
            print(f"  • {fb}")
        print(f"  ➜ Requête finale reformulée : \"{final_state.get('current_query')}\"")
        print("-" * 85)

    # 3. Verdict Global et Synthèse
    cb = final_state.get("confidence_breakdown", {})
    weights = cb.get("weights", {})

    print("\n" + "=" * 85)
    print(f"⚖️ VERDICT FINAL : {report_dict.get('verdict_fr', 'Inconnu')}")
    print(
        f"• Score de certitude composite  : {report_dict.get('confidence_score', 0.0) * 100:.1f}%"
    )
    print(
        f"• Équilibre des sources         : {report_dict.get('supports_count', 0)} Soutient | {report_dict.get('refutes_count', 0)} Contredit | {report_dict.get('neutral_count', 0)} Neutre"
    )
    print(
        f"• Tours d'exécution             : {final_state.get('iteration_count', 0) + 1}"
    )
    print("=" * 85)

    print("\n📊 DÉCOMPOSITION DU SCORE ÉPISTÉMIQUE :")
    print("-" * 85)
    print(
        f"• Accord & Consensus  (Poids {weights.get('consensus', 0.35)*100:.0f}%) : {cb.get('score_consensus', 0.0)*100:.1f}%"
    )
    print(
        f"• Reranker Sémantique (Poids {weights.get('reranker', 0.25)*100:.0f}%) : {cb.get('score_reranker', 0.0)*100:.1f}%"
    )
    print(
        f"• Déduction NLI       (Poids {weights.get('nli', 0.25)*100:.0f}%) : {cb.get('score_nli', 0.0)*100:.1f}%"
    )
    print(
        f"• Récence Temporelle  (Poids {weights.get('recency', 0.15)*100:.0f}%) : {cb.get('score_recency', 0.0)*100:.1f}%"
    )
    print(f"➜ Justification : {cb.get('explanation', '')}")
    print("-" * 85)

    # 4. Détail des preuves et audits textuels
    evidences = report_dict.get("evidences", [])
    print(
        f"\n📑 PREUVES VALIDÉES PAR L'AGENT VÉRIFICATEUR ({len(evidences)} extraits) :"
    )
    print("=" * 85)

    for i, ev in enumerate(evidences, start=1):
        stance = ev.get("nli_stance", {})
        label = stance.get("label", "NEUTRAL")
        conf = stance.get("confidence", 0.0) * 100

        badge = (
            "🟢 SOUTIENT"
            if label == "SUPPORTS"
            else ("🔴 CONTREDIT" if label == "REFUTES" else "⚪ NEUTRE")
        )

        print(f"\n--- PREUVE #{i} : {badge} ({conf:.1f}% confiance NLI) ---")
        print(f"• Titre       : {ev.get('title')}")
        print(
            f"• arXiv ID    : {ev.get('paper_id')} ({ev.get('published_date')}) | Catégorie: {ev.get('primary_category')}"
        )
        print(f"• URL arXiv   : {ev.get('entry_id')}")
        print(
            f"• Rerank      : {ev.get('rerank_score', 0.0)*100:.1f}% (Rang initial: #{ev.get('original_hybrid_rank')} ➜ #{ev.get('final_rank')})"
        )

        audit = ev.get("citation_check", {})
        audit_badge = (
            "✅ VALIDÉ MOT POUR MOT"
            if audit.get("is_valid")
            else "❌ ALIEN / NON TROUVÉ"
        )
        print(f"• Audit textuel : {audit_badge}")

        if args.verbose:
            print(f"• Extrait brut analysé :")
            print(f"  \"{ev.get('raw_content', '')[:280]}...\"")

    print("\n" + "=" * 85 + "\n")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(report_dict, indent=2, ensure_ascii=False))
        print(f"[OK] Rapport officiel sauvegardé dans : {out_p.resolve()}")


if __name__ == "__main__":
    main()
