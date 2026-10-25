// Adaptation de la réponse de /api/verify vers le modèle de vue de l'interface.

export const fmt = (n) => (Number.isInteger(n) ? String(n) : n.toFixed(1)).replace('.', ',');

/** Pastille d'état du LLM dans l'en-tête, d'après /api/stats. */
export function llmPill(stats) {
  const llm = stats?.llm;
  if (!llm) return { label: 'LLM : état inconnu', warn: true };
  if (llm.enabled) return { label: `LLM : ${llm.provider} / ${llm.model}`, warn: false };
  const keyMissing = /clé api/i.test(llm.reason || '');
  return { label: keyMissing ? 'LLM : clé API absente' : 'LLM : désactivé', warn: true };
}

export const AGENTS = [
  { name: 'Analyseur', node: 'analyseur', desc: 'Normalise l’affirmation et dérive des requêtes de recherche.' },
  { name: 'Récupérateur', node: 'recuperateur', desc: 'Recherche hybride BM25 + FAISS dans le corpus.' },
  { name: 'Évaluateur', node: 'evaluateur', desc: 'Reranking Cross-Encoder, puis NLI DeBERTa-v3 par passage.' },
  { name: 'Générateur', node: 'generateur', desc: 'Rédige une réponse ancrée, avec citations.' },
  { name: 'Vérificateur', node: 'verificateur', desc: 'Audite chaque citation dans l’abstract stocké.' },
  { name: 'Synthèse', node: 'synthese', desc: 'Agrège les scores et rend le verdict.' },
];

const VERDICTS = {
  SOUTENU: { tone: 'sup', label: 'Soutenu', read: 'Consensus, reranker et NLI convergent vers le soutien de l’affirmation.' },
  REFUTE: { tone: 'con', label: 'Réfuté', read: 'La majorité des passages pertinents contredit l’affirmation.' },
  SUJET_DEBATTU: { tone: 'deb', label: 'Sujet débattu', read: 'Les passages pertinents se partagent entre soutien et contradiction.' },
  PREUVES_INSUFFISANTES: { tone: 'ins', label: 'Preuves insuffisantes', read: 'Trop peu de passages pertinents pour conclure dans un sens ou dans l’autre.' },
};

const NLI_TONE = { SUPPORTS: 'sup', REFUTES: 'con', NEUTRAL: 'neu' };
const NLI_LABEL = { sup: 'SOUTIENT', con: 'CONTREDIT', neu: 'NEUTRE' };

const LLM_STANCES = {
  SUPPORTS: { label: 'SOUTIENT', tone: 'tone-sup', shape: 'sh-sup' },
  REFUTES: { label: 'CONTREDIT', tone: 'tone-con', shape: 'sh-con' },
  MIXED: { label: 'SOURCES PARTAGÉES', tone: 'tone-deb', shape: 'sh-neu' },
  INSUFFICIENT: { label: 'INSUFFISANT', tone: 'tone-ins', shape: 'sh-off' },
};

/** Titre court pour les étiquettes du graphe. */
function shortTitle(title) {
  const t = (title || '').split(':')[0].trim();
  return t.length > 24 ? `${t.slice(0, 23).trimEnd()}…` : t;
}

function llmViewModel(data, verdictLabel) {
  const info = data.llm || null;
  const answer = data.answer || null;
  const status = info?.status || (answer ? 'OK' : 'DISABLED');

  if (answer) {
    const st = LLM_STANCES[answer.stance] || LLM_STANCES.INSUFFICIENT;
    return {
      mode: 'ok',
      model: info ? `${info.model} · ${info.provider}` : '',
      provider: info?.provider || '',
      latency: info?.latency_s != null ? `Latence ${fmt(info.latency_s)} s` : '',
      stance: st,
      answer: answer.text,
      quotes: (answer.citations || []).map((c) => ({
        text: c.quote,
        paper: c.title || c.chunk_id,
        ok: !!c.is_verified,
      })),
      stanceBanner: answer.agrees_with_nli === false
        ? `LLM : ${st.label.toLowerCase()} · verdict calculé : ${verdictLabel.toLowerCase()}. Fiez-vous aux passages et aux scores plutôt qu’à la seule réponse rédigée.`
        : null,
      quoteBanner: answer.citations_verified === false,
    };
  }
  if (status === 'DISABLED') {
    const keyMissing = /clé api/i.test(info?.message || '');
    return { mode: 'disabled', title: keyMissing ? 'Clé API absente' : 'LLM désactivé', message: info?.message || '', keyMissing };
  }
  if (status === 'SKIPPED') return { mode: 'skipped', message: info?.message || '' };
  return { mode: 'error', message: info?.message || 'Réponse du LLM indisponible.' };
}

function evidenceViewModel(ev, i, citations) {
  const stance = ev.nli_stance || {};
  const st = NLI_TONE[stance.label] || 'neu';
  const nli = (stance.confidence || 0) * 100;
  const rr = (ev.rerank_score || 0) * 100;
  const from = ev.original_hybrid_rank ?? i + 1;
  const to = ev.final_rank ?? i + 1;

  // Audit : la citation du LLM sur ce passage s'il y en a une, sinon la phrase retenue comme preuve.
  const llmCites = citations.filter((c) => c.chunk_id === ev.chunk_id);
  let audit;
  if (llmCites.length) {
    const found = llmCites.every((c) => c.is_verified);
    audit = {
      state: found ? 'found' : 'missing',
      pill: found ? 'Citation retrouvée' : 'Non retrouvée',
      text: found
        ? 'Le passage cité par le LLM figure dans l’abstract stocké.'
        : 'La citation du LLM n’apparaît pas à l’identique dans l’abstract.',
    };
  } else if (ev.citation_check) {
    const found = !!ev.citation_check.is_valid;
    audit = {
      state: found ? 'found' : 'missing',
      pill: found ? 'Extrait retrouvé' : 'Non retrouvé',
      text: found
        ? 'La phrase retenue comme preuve figure à l’identique dans l’abstract stocké.'
        : 'La phrase retenue n’apparaît pas à l’identique dans l’abstract stocké.',
    };
  } else {
    audit = { state: 'none', pill: 'Non cité', text: 'Cet extrait n’a pas été cité par le LLM.' };
  }

  const quote = ev.evidence_quote || ev.raw_content || '';
  return {
    st,
    rank: `#${i + 1}`,
    rankNo: i + 1,
    stanceLabel: NLI_LABEL[st],
    title: ev.title,
    short: shortTitle(ev.title),
    date: ev.published_date,
    cat: ev.primary_category,
    idText: ev.paper_id,
    arxiv: ev.entry_id || `https://arxiv.org/abs/${ev.paper_id}`,
    pdf: ev.pdf_url || `https://arxiv.org/pdf/${ev.paper_id}`,
    quote,
    fullText: ev.raw_content && ev.raw_content !== quote ? ev.raw_content : null,
    nli,
    rr,
    from,
    to,
    rankCls: to < from ? 'up' : to > from ? 'down' : '',
    audit,
  };
}

/**
 * @param data     réponse JSON de l'API
 * @param tries    nombre d'essais max choisi par l'utilisateur
 * @param elapsed  durée mesurée côté client (secondes)
 */
export function buildReport(data, tries, elapsed) {
  const v = VERDICTS[data.verdict] || VERDICTS.PREUVES_INSUFFISANTES;
  const citations = data.answer?.citations || [];
  const evidence = (data.evidences || []).map((e, i) => evidenceViewModel(e, i, citations));
  const counts = [data.supports_count || 0, data.refutes_count || 0, data.neutral_count || 0];
  const kept = Math.max(data.total_evidence_evaluated || 0, evidence.length);
  const disc = data.discarded_passages || 0;

  // Sans aucun passage retenu, une confiance chiffrée n'aurait aucun sens.
  const hasEvidence = kept > 0;
  const b = data.confidence_breakdown;
  const conf = hasEvidence ? (data.confidence_score || 0) * 100 : null;
  const dimValues = hasEvidence && b
    ? [b.score_consensus, b.score_reranker, b.score_nli, b.score_recency].map((x) => x * 100)
    : [null, null, null, null];
  const weights = b?.weights || {};
  const dims = [
    ['Consensus', weights.consensus ?? 0.35],
    ['Cross-Encoder', weights.reranker ?? 0.25],
    ['DeBERTa', weights.nli ?? 0.25],
    ['Récence', weights.recency ?? 0.15],
  ].map(([name, w], i) => ({
    name,
    weight: Math.round(w * 100),
    value: dimValues[i] == null ? '—' : fmt(dimValues[i]),
    width: dimValues[i] == null ? '0%' : `${Math.max(0, Math.min(100, dimValues[i]))}%`,
  }));

  const llm = llmViewModel(data, v.label);
  const plural = (n, one, many) => `${n} ${n > 1 ? many : one}`;
  const metaLine = [
    plural(kept + disc, 'passage récupéré', 'passages récupérés'),
    plural(kept, 'retenu', 'retenus'),
    plural(disc, 'écarté', 'écartés'),
    `essai ${data.iteration_count || 1}/${tries}`,
    elapsed != null ? `${fmt(elapsed)} s` : null,
  ].filter(Boolean).join(' · ');

  return {
    claim: data.claim,
    tone: v.tone,
    label: v.label,
    read: hasEvidence ? v.read : 'Aucun passage du corpus ne dépasse le seuil de pertinence.',
    conf,
    confNote: conf == null
      ? 'Aucune confiance calculée : aucun passage n’a franchi le seuil de pertinence.'
      : 'Indice composite pondéré, pas une probabilité que l’affirmation soit vraie.',
    explanation: hasEvidence ? b?.explanation : null,
    dims,
    counts,
    kept,
    disc,
    metaLine,
    evidence,
    llm,
    logs: data.agent_logs || [],
    feedback: data.feedback_history || [],
    raw: data,
  };
}
