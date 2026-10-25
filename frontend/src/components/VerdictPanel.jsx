import { fmt } from '../lib/report.js';

export default function VerdictPanel({ report }) {
  const [sup, con, neu] = report.counts;
  return (
    <section className={`card xh tone-${report.tone}`} aria-label="Verdict">
      <span className="cap">[VERDICT]</span>
      <h2 className="vword">{report.label}</h2>
      <p className="vread">{report.read}</p>
      <div className="fig-row">
        <div>
          <span className="cap">[CONFIDENCE]</span>
          <div className="fig">{report.conf == null ? '—' : fmt(report.conf)}<small>%</small></div>
        </div>
      </div>
      <p className="fig-note">{report.confNote}</p>
      <ul className="dims" aria-label="Décomposition de la confiance">
        {report.dims.map((d) => (
          <li key={d.name}>
            <div className="dim-top">
              <span className="dim-name">{d.name}</span>
              <span className="cap">× {d.weight} %</span>
              <span className="dim-val">{d.value}</span>
            </div>
            <div className="bar"><i style={{ width: d.width }} /></div>
          </li>
        ))}
      </ul>
      {report.explanation && <p className="fig-note" style={{ marginTop: 18 }}>{report.explanation}</p>}
      <div className="counts" aria-label="Répartition des passages retenus">
        <span className="tone-sup"><i className="sh sh-sup" />{sup} soutiennent</span>
        <span className="tone-con"><i className="sh sh-con" />{con} contredisent</span>
        <span className="tone-neu"><i className="sh sh-neu" />{neu} neutres</span>
      </div>
    </section>
  );
}
