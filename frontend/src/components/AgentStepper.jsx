import Icon from './Icon.jsx';
import { AGENTS } from '../lib/report.js';

const STEP_H = 84; // hauteur d'une ligne (cf. .step)

/** Flèche de la boucle de rétroaction : de l'Évaluateur (2) vers l'Analyseur (0). */
function loopPaths(from, to) {
  const y1 = from * STEP_H + STEP_H / 2;
  const y0 = to * STEP_H + STEP_H / 2;
  return {
    line: `M6 ${y1}C54 ${y1} 54 ${y0} 18 ${y0}`,
    head: `M26 ${y0 - 7}L16 ${y0}L26 ${y0 + 7}`,
  };
}

export default function AgentStepper({ claim, step, retry, tries, llm, lastFeedback, onCancel }) {
  const loop = loopPaths(2, 0);
  const rows = AGENTS.map((a, i) => {
    const done = i < step;
    const active = i === step;
    // Après une rétroaction, l'Analyseur, le Récupérateur et l'Évaluateur sont rejoués.
    const redo = retry > 0 && i <= 2;
    let desc = a.desc;
    if (a.node === 'generateur' && llm && !llm.enabled) desc = 'LLM non configuré : étape ignorée.';
    return {
      n: `0${i + 1}`,
      name: a.name,
      desc,
      cls: done ? 'is-done' : active ? 'is-active' : 'is-wait',
      state: done ? 'terminé' : active ? (redo ? `essai ${retry + 1}/${tries}` : 'en cours') : 'en attente',
      current: active ? 'step' : undefined,
    };
  });

  return (
    <section className="card xh" aria-label="Progression des agents">
      <div className="sec-h" style={{ marginBottom: 0 }}>
        <span className="cap">[PIPELINE] Agent {Math.min(step, 5) + 1}/06</span>
        <button className="btn btn-sm" type="button" onClick={onCancel} style={{ marginLeft: 'auto' }}>Annuler</button>
      </div>
      <p className="claim-recap" lang="en">{claim}</p>
      <ol className="steps">
        {rows.map((s) => (
          <li key={s.n} className={`step ${s.cls}`} aria-current={s.current}>
            <span className="step-n">{s.n}<small>/06</small></span>
            <div>
              <div className="step-name">{s.name}</div>
              <div className="step-desc">{s.desc}</div>
            </div>
            <span className="step-state">{s.state}</span>
          </li>
        ))}
        <svg className={`loop ${retry > 0 ? 'is-on' : ''}`} viewBox="0 0 54 504" aria-hidden="true">
          <path className="loop-line" d={loop.line} />
          <path className="loop-head" d={loop.head} />
        </svg>
      </ol>
      {retry > 0 && (
        <div className="retry" role="status">
          <Icon name="refresh" />
          <div>
            <b>[BOUCLE_DE_RÉTROACTION]</b>
            {lastFeedback || 'Preuves insuffisantes : l’Analyseur reformule la recherche.'}
          </div>
        </div>
      )}
      <span className="sr-only" role="status" aria-live="polite">
        Étape {Math.min(step, 5) + 1} sur 6 : {AGENTS[Math.min(step, 5)].name}
      </span>
    </section>
  );
}
