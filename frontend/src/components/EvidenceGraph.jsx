import { useMemo } from 'react';
import { buildGeometry, buildView, seedFrom } from '../lib/graph.js';

const HERO_SPEC = { seed: 11, top: [], counts: [8, 5, 7], disc: 9, sx: 1.3, sy: 1 };
const LOAD_SPEC = { seed: 29, top: [], counts: [4, 2, 3], disc: 3, sx: 1.05, sy: 0.9 };

/**
 * mode 'hero'   : décor de l'accueil (réseau illustratif)
 * mode 'load'   : construction pendant l'exécution (passages « en attente »)
 * mode 'result' : graphe construit à partir du rapport réel
 */
export default function EvidenceGraph({ mode, step = 0, agent, report, corpusLabel }) {
  const geo = useMemo(() => {
    if (mode === 'hero') return buildGeometry(HERO_SPEC);
    if (mode === 'load') return buildGeometry(LOAD_SPEC);
    return buildGeometry({
      seed: seedFrom(report.claim || ''),
      top: report.evidence.map((e) => ({ st: e.st, rel: Math.max(0.3, Math.min(1, e.rr / 100)) })),
      counts: report.counts,
      disc: report.disc,
      sx: 1.05,
      sy: 0.9,
    });
  }, [mode, report]);

  const g = useMemo(
    () => buildView(geo, mode, step, mode === 'result' ? report : { agent, corpusLabel }),
    [geo, mode, step, agent, report, corpusLabel],
  );

  return (
    <div className="gbox">
      <svg viewBox="0 0 1000 680" role="img" aria-label={g.aria}>
        <g className="gl-sway">
          <g className="gl-twigs">
            {g.twigs.map((t, i) => (
              <g key={i} className={t.cls} style={{ transitionDelay: t.td }}>
                <path className="gl-twig" d={t.d} strokeDasharray={t.dash} strokeDashoffset={t.off} />
                <circle className="gl-tip" cx={t.x} cy={t.y} r={t.r} />
              </g>
            ))}
          </g>
          <g className="gl-nodes">
            {g.nodes.map((n, i) => (
              <g key={i} className={n.cls} style={{ transitionDelay: n.td }}>
                <path className="gl-branch" d={n.d} strokeWidth={n.w} strokeDasharray={n.dash} strokeDashoffset={n.off} />
                <circle className="gl-halo" cx={n.hx} cy={n.hy} r={n.hr} style={{ animationDelay: n.delay }} />
                <path className="gl-shape" d={n.shape} />
              </g>
            ))}
          </g>
          <g className={g.centerCls}>
            <circle className="gl-core-halo2" cx="500" cy="340" r="66" />
            <circle className="gl-core-halo" cx="500" cy="340" r="36" />
            <circle className="gl-ring-track" cx="500" cy="340" r="40" />
            <circle className="gl-ring-arc" cx="500" cy="340" r="40" strokeDasharray={g.ringDash} transform="rotate(-90 500 340)" />
            <circle className="gl-core" cx="500" cy="340" r="13" />
            <path className="gl-cross" d="M500 292V280M500 388V400M452 340H440M548 340H560" />
          </g>
          <g className="gl-callouts">
            {g.callouts.map((c) => (
              <g key={c.id}>
                <path className="gl-co-line" d={c.d} />
                <circle className="gl-co-pt" cx={c.tx} cy={c.ty} r="9" />
              </g>
            ))}
          </g>
        </g>
      </svg>
      {g.callouts.map((c) => (
        <div key={c.id} className={`co ${c.cls}`} style={{ left: c.left, top: c.top }}>
          <span className="co-id">{c.id}</span>
          <span className="co-sub" lang={c.id.startsWith('[EVIDENCE') ? 'en' : undefined}>{c.sub}</span>
        </div>
      ))}
    </div>
  );
}

export function GraphLegend() {
  return (
    <ul className="legend" aria-label="Légende du graphe d’évidence">
      <li className="tone-sup"><i className="sh sh-sup" /><span style={{ color: 'var(--text-2)' }}>Soutient</span></li>
      <li className="tone-con"><i className="sh sh-con" /><span style={{ color: 'var(--text-2)' }}>Contredit</span></li>
      <li className="tone-neu"><i className="sh sh-neu" /><span style={{ color: 'var(--text-2)' }}>Neutre</span></li>
      <li><i className="sh sh-off" />Écarté (sous le seuil)</li>
      <li className="cap">Épaisseur du trait = pertinence</li>
    </ul>
  );
}
