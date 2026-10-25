import Icon from './Icon.jsx';
import { fmt } from '../lib/report.js';

function ExternalLink({ href, children }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
      <Icon name="external" />
    </a>
  );
}

function Card({ e }) {
  return (
    <article className={`card ev tone-${e.st}`}>
      <div className="ev-top">
        <span className="ev-rank" aria-label={`Rang ${e.rankNo}`}>{e.rank}</span>
        <span className="pill pill-tone"><i className={`sh sh-${e.st}`} />{e.stanceLabel} {fmt(e.nli)} %</span>
      </div>
      <h3 lang="en">{e.title}</h3>
      <div className="ev-meta">
        <span>{e.date}</span>
        <span className="pill">{e.cat}</span>
        <ExternalLink href={e.arxiv}>arXiv {e.idText}</ExternalLink>
        <ExternalLink href={e.pdf}>PDF</ExternalLink>
      </div>
      <blockquote lang="en">{e.quote}</blockquote>
      {e.fullText && (
        <details className="ev-more">
          <summary className="cap">Extrait complet</summary>
          <p lang="en">{e.fullText}</p>
        </details>
      )}
      <div className="meters">
        <div>
          <div className="meter-top"><span className="cap">NLI · DeBERTa-v3</span><b>{fmt(e.nli)} %</b></div>
          <div className="bar"><i style={{ width: `${e.nli}%` }} /></div>
        </div>
        <div>
          <div className="meter-top"><span className="cap">Rerank · Cross-Encoder</span><b>{fmt(e.rr)} %</b></div>
          <div className="bar"><i style={{ width: `${e.rr}%` }} /></div>
        </div>
        <div className={`rank ${e.rankCls}`} aria-label={`Rang passé de ${e.from} à ${e.to}`}>
          <span className="cap">Rang</span>#{e.from} ➔ #{e.to}
        </div>
      </div>
      <div className="audit">
        {e.audit.state === 'found' && (
          <span className="pill pill-tone tone-sup"><Icon name="check" size={14} />{e.audit.pill}</span>
        )}
        {e.audit.state === 'missing' && (
          <span className="pill pill-tone tone-warn"><Icon name="cross" size={14} />{e.audit.pill}</span>
        )}
        {e.audit.state === 'none' && <span className="pill">{e.audit.pill}</span>}
        <span>{e.audit.text}</span>
      </div>
    </article>
  );
}

export default function EvidenceCards({ report }) {
  return (
    <section className="sec" aria-labelledby="ev-title">
      <div className="sec-h">
        <span className="cap">[PREUVES]</span>
        <h2 id="ev-title" className="h2">Passages les mieux classés</h2>
      </div>
      {report.evidence.length > 0 ? (
        <div className="evs">
          {report.evidence.map((e) => <Card key={`${e.idText}-${e.rankNo}`} e={e} />)}
        </div>
      ) : (
        <div className="empty-ev" role="status">
          <span className="cap">[AUCUN_PASSAGE]</span>
          <h3 className="h2" style={{ marginTop: 8 }}>Aucun passage pertinent</h3>
          <p style={{ margin: '10px 0 0', color: 'var(--text-2)', maxWidth: '60ch' }}>
            {report.disc > 0
              ? `Les ${report.disc} passages récupérés sont tous passés sous le seuil de pertinence du Cross-Encoder. `
              : 'Aucun passage n’a été trouvé dans le corpus. '}
            Reformulez l’affirmation, ou vérifiez que le sujet est couvert par les articles indexés.
          </p>
        </div>
      )}
    </section>
  );
}
