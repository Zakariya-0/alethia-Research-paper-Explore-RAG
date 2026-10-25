import Icon from './Icon.jsx';

function Notice({ provider }) {
  return (
    <p className="notice notice-llm">
      <Icon name="globe" />
      <span>
        L’affirmation et les passages sélectionnés ont été envoyés à une API distante ({provider}). L’audit vérifie
        la présence de chaque citation dans l’abstract stocké ; il ne garantit pas que la réponse soit exacte.
      </span>
    </p>
  );
}

function WarnBanner({ title, children, style }) {
  return (
    <div className="banner banner-warn" role="status" style={style}>
      <Icon name="warn" />
      <div><b>{title}</b><span>{children}</span></div>
    </div>
  );
}

export default function LlmAnswer({ llm, onRetry }) {
  return (
    <section className="sec" aria-labelledby="llm-title">
      <div className="sec-h">
        <span className="cap">[RÉPONSE_LLM]</span>
        <h2 id="llm-title" className="h2">Réponse rédigée à partir des passages</h2>
      </div>
      <div className="card xh">
        {llm.mode === 'ok' && (
          <>
            <div className="llm-top">
              <span className="pill"><Icon name="chip" size={14} />{llm.model}</span>
              {llm.latency && <span className="lat">{llm.latency}</span>}
              <span className={`pill pill-tone ${llm.stance.tone}`}><i className={`sh ${llm.stance.shape}`} />{llm.stance.label}</span>
            </div>
            {llm.stanceBanner && (
              <WarnBanner title="La position du LLM diffère du verdict calculé par DeBERTa" style={{ marginBottom: 12 }}>
                {llm.stanceBanner}
              </WarnBanner>
            )}
            {llm.quoteBanner && (
              <WarnBanner title="Certaines citations n’ont pas été retrouvées à l’identique dans les abstracts" style={{ marginBottom: 20 }}>
                Elles sont conservées et signalées ci-dessous ; vérifiez-les dans l’article source.
              </WarnBanner>
            )}
            <p className="answer">{llm.answer}</p>
            {llm.quotes.length > 0 ? (
              <ul className="quotes" aria-label="Citations de la réponse">
                {llm.quotes.map((q, i) => (
                  <li className="quote" key={i}>
                    <div>
                      {q.ok ? (
                        <span className="pill pill-tone tone-sup"><Icon name="check" size={14} />Vérifiée</span>
                      ) : (
                        <span className="pill pill-tone tone-warn"><Icon name="cross" size={14} />Non retrouvée</span>
                      )}
                    </div>
                    <q lang="en">{q.text}</q>
                    <span className="paper" lang="en">{q.paper}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="hint">Aucune citation produite : rien à auditer.</p>
            )}
            <Notice provider={llm.provider} />
          </>
        )}

        {llm.mode === 'disabled' && (
          <div className="state-card tone-warn">
            <Icon name={llm.keyMissing ? 'key' : 'skip'} style={{ color: 'var(--warn)' }} />
            <div>
              <b>{llm.title}</b>
              <span>
                Génération désactivée : le verdict ci-dessus provient uniquement de la recherche, du reranker et du NLI.
                {llm.message ? ` ${llm.message}.` : ''}
              </span>
            </div>
          </div>
        )}

        {llm.mode === 'error' && (
          <div className="state-card tone-warn" role="alert">
            <Icon name="warn" style={{ color: 'var(--err)' }} />
            <div>
              <b>Erreur LLM</b>
              <span>{llm.message} Le verdict calculé n’est pas affecté ; aucune citation n’a pu être auditée.</span>
            </div>
            <button className="btn btn-sm" type="button" onClick={onRetry}>
              <Icon name="refresh" />Relancer la vérification
            </button>
          </div>
        )}

        {llm.mode === 'skipped' && (
          <div className="state-card">
            <Icon name="skip" style={{ color: 'var(--text-3)' }} />
            <div>
              <b>Génération ignorée</b>
              <span>Aucun passage ne dépasse le seuil de pertinence : le LLM n’a rien à citer, il n’a donc pas été appelé.</span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
