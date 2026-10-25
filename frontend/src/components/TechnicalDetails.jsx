import { useState } from 'react';
import Icon from './Icon.jsx';

function Collapsible({ id, tag, title, count, children }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="coll">
      <button className="coll-h" type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}>
        <span className="cap">{tag}</span>
        <span>{title}</span>
        <span className="count">{count}</span>
        <Icon name="chevron" className={`ic ${open ? 'rot' : ''}`} />
      </button>
      {open && <div id={id}>{children}</div>}
    </div>
  );
}

export default function TechnicalDetails({ report }) {
  const logs = report.logs;

  return (
    <section className="sec" aria-label="Détails techniques">
      <Collapsible id="log-body" tag="[JOURNAL_AGENTS]" title="Journal d’audit des agents" count={`${report.logs.length} événements`}>
        <div className="coll-b scroll-x">
          <div className="lg" role="table" aria-label="Journal d’audit des agents">
            <div className="lg-row lg-head" role="row">
              <span className="cap" role="columnheader">Heure UTC</span>
              <span className="cap" role="columnheader">Agent</span>
              <span className="cap" role="columnheader">Action</span>
              <span className="cap" role="columnheader">Détails</span>
            </div>
            {logs.map((r, i) => (
              <div className="lg-row" role="row" key={i}>
                <span className="lg-t" role="cell">{r.timestamp}</span>
                <span className="lg-a" role="cell">{r.agent}</span>
                <span className="lg-x" role="cell">{r.action}</span>
                <span className="lg-d" role="cell">{r.details}</span>
              </div>
            ))}
          </div>
        </div>
      </Collapsible>
      <Collapsible id="json-body" tag="[JSON]" title="Réponse brute de l’API" count="lecture seule">
        <div className="coll-b">
          <pre className="json" tabIndex={0} aria-label="Réponse JSON">{JSON.stringify(report.raw, null, 2)}</pre>
        </div>
      </Collapsible>
    </section>
  );
}
