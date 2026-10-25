// Géométrie du graphe d'évidence : réseau déterministe de passages autour de l'affirmation.
// Port du prototype Claude Design : fonctions pures, alimentées par les vraies données.

const CX = 500;
const CY = 340;
const RING_LEN = 251.33;

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Graine entière stable à partir d'un texte (même affirmation = même graphe). */
export function seedFrom(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function shuffle(arr, rand) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

function bez(p0, p1, p2, p3, t) {
  const u = 1 - t;
  return {
    x: u * u * u * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t * t * t * p3.x,
    y: u * u * u * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t * t * t * p3.y,
  };
}

function twig(x, y, ang, L, rand, side) {
  const cx = x + Math.cos(ang - side * 0.3) * L * 0.55;
  const cy = y + Math.sin(ang - side * 0.3) * L * 0.55;
  const ex = x + Math.cos(ang) * L;
  const ey = y + Math.sin(ang) * L;
  return {
    d: `M${x.toFixed(1)} ${y.toFixed(1)}Q${cx.toFixed(1)} ${cy.toFixed(1)} ${ex.toFixed(1)} ${ey.toFixed(1)}`,
    len: Math.round(L * 1.12),
    x: +ex.toFixed(1),
    y: +ey.toFixed(1),
    r: +(1.2 + rand() * 1.5).toFixed(1),
  };
}

/**
 * Construit les nœuds du graphe.
 * spec = { seed, top: [{st, rel}], counts: [sup, con, neu], disc, sx, sy }
 *  - top    : passages affichés en cartes (stance réelle, pertinence réelle du Cross-Encoder)
 *  - counts : passages retenus au total (le reste est complété à partir de ces compteurs)
 *  - disc   : passages écartés sous le seuil de pertinence
 */
export function buildGeometry(spec) {
  const rand = rng(spec.seed);
  const items = [];
  spec.top.forEach((t) => items.push({ st: t.st, rel: t.rel }));

  const rest = { sup: spec.counts[0], con: spec.counts[1], neu: spec.counts[2] };
  spec.top.forEach((t) => {
    rest[t.st] = Math.max(0, rest[t.st] - 1);
  });
  ['sup', 'con', 'neu'].forEach((st) => {
    for (let k = 0; k < rest[st]; k++) items.push({ st, rel: 0.42 + rand() * 0.36 });
  });
  for (let k = 0; k < spec.disc; k++) items.push({ st: 'off', rel: 0.05 + rand() * 0.18 });

  const N = items.length;
  const order = shuffle(Array.from({ length: N }, (_, i) => i), rand);
  const nodes = [];

  items.forEach((it, i) => {
    const off = it.st === 'off';
    const a = ((order[i] + 0.5) / N) * Math.PI * 2 + (rand() - 0.5) * ((Math.PI * 2) / N) * 0.6;
    const R = off ? 250 + rand() * 80 : 140 + (1 - it.rel) * 165 + rand() * 26;
    let x = CX + Math.cos(a) * R * spec.sx;
    let y = CY + Math.sin(a) * R * spec.sy;
    x = Math.max(70, Math.min(930, x));
    y = Math.max(50, Math.min(630, y));

    const p0 = { x: CX, y: CY };
    const dx = x - CX;
    const dy = y - CY;
    const L = Math.hypot(dx, dy);
    const px = -dy / L;
    const py = dx / L;
    const k1 = (rand() - 0.5) * L * 0.36;
    const k2 = (rand() - 0.5) * L * 0.3;
    const c1 = { x: CX + dx * 0.33 + px * k1, y: CY + dy * 0.33 + py * k1 };
    const c2 = { x: CX + dx * 0.7 + px * k2, y: CY + dy * 0.7 + py * k2 };
    const p3 = { x, y };
    const d = `M${CX} ${CY}C${c1.x.toFixed(1)} ${c1.y.toFixed(1)} ${c2.x.toFixed(1)} ${c2.y.toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)}`;

    let len = 0;
    let prev = p0;
    for (let s = 1; s <= 16; s++) {
      const q = bez(p0, c1, c2, p3, s / 16);
      len += Math.hypot(q.x - prev.x, q.y - prev.y);
      prev = q;
    }

    const tw = [];
    const nt = off ? 1 : 3 + Math.floor(rand() * 3);
    for (let m = 0; m < nt; m++) {
      const t0 = 0.4 + rand() * 0.5;
      const b = bez(p0, c1, c2, p3, t0);
      const b2 = bez(p0, c1, c2, p3, Math.min(1, t0 + 0.03));
      const ang = Math.atan2(b2.y - b.y, b2.x - b.x);
      const side = rand() < 0.5 ? -1 : 1;
      tw.push(twig(b.x, b.y, ang + side * (0.45 + rand() * 0.9), (off ? 18 : 26 + rand() * 46) * (0.7 + it.rel * 0.6), rand, side));
    }
    if (!off) {
      const out = Math.atan2(dy, dx);
      for (let m = 0; m < 3; m++) {
        tw.push(twig(x, y, out + (m - 1) * 0.55 + (rand() - 0.5) * 0.2, 16 + rand() * 28, rand, m === 0 ? -1 : 1));
      }
    }

    nodes.push({
      x, y, st: it.st, rel: it.rel, d, len, tw,
      r: off ? 3.4 : 4.6 + it.rel * 5.4,
      w: off ? 0.8 : 0.9 + it.rel * 3,
      pulse: !off && rand() < 0.3,
      delay: `${(rand() * 4).toFixed(2)}s`,
    });
  });

  return { nodes };
}

/** Répartit les étiquettes de rappel sur les bords gauche/droit sans chevauchement. */
function place(items) {
  const out = [];
  ['l', 'r'].forEach((sd) => {
    const arr = items.filter((it) => it.side === sd).sort((a, b) => a.y - b.y);
    let prev = -1e9;
    arr.forEach((it) => {
      it.yy = Math.max(it.y, 55, prev + 74);
      prev = it.yy;
    });
    if (arr.length) {
      const over = arr[arr.length - 1].yy - 625;
      if (over > 0) arr.forEach((it) => { it.yy -= over; });
    }
    arr.forEach((it) => {
      const lx = sd === 'l' ? 190 : 810;
      const kx = sd === 'l' ? Math.max(lx + 12, it.tx - 36) : Math.min(lx - 12, it.tx + 36);
      out.push({
        id: it.id,
        sub: it.sub,
        cls: sd === 'l' ? 'al-l' : 'al-r',
        left: sd === 'l' ? '3.5%' : '96.5%',
        top: `${(it.yy / 6.8).toFixed(2)}%`,
        d: `M${lx} ${it.yy.toFixed(1)}L${kx.toFixed(1)} ${it.yy.toFixed(1)}L${it.tx.toFixed(1)} ${it.ty.toFixed(1)}`,
        tx: it.tx,
        ty: it.ty,
      });
    });
  });
  return out;
}

const fmtPct = (n) => (Number.isInteger(n) ? String(n) : n.toFixed(1)).replace('.', ',');

/**
 * Vue SVG d'une géométrie.
 *  mode 'hero'   : décor de l'écran d'accueil
 *  mode 'load'   : construction pendant l'exécution (nœuds « en attente », sans stance inventée)
 *  mode 'result' : graphe des vraies données (ctx : label, tone, conf, kept, disc, counts, evidence[])
 */
export function buildView(geo, mode, step, ctx = {}) {
  const nodes = [];
  const twigs = [];

  geo.nodes.forEach((n, i) => {
    let kind = n.st;
    let vis = true;
    if (mode === 'load') {
      if (step < 1) vis = false;
      kind = 'pend';
    }
    const off = kind === 'off';
    const w = mode === 'load' ? 0.8 : n.w;
    const r = kind === 'pend' ? n.r * 0.75 : n.r;
    let shape;
    if (kind === 'con') {
      const q = r * 1.35;
      shape = `M${n.x.toFixed(1)} ${(n.y - q).toFixed(1)}L${(n.x + q).toFixed(1)} ${n.y.toFixed(1)}L${n.x.toFixed(1)} ${(n.y + q).toFixed(1)}L${(n.x - q).toFixed(1)} ${n.y.toFixed(1)}Z`;
    } else {
      shape = `M${(n.x - r).toFixed(1)} ${n.y.toFixed(1)}a${r.toFixed(1)} ${r.toFixed(1)} 0 1 0 ${(2 * r).toFixed(1)} 0a${r.toFixed(1)} ${r.toFixed(1)} 0 1 0 ${(-2 * r).toFixed(1)} 0`;
    }
    const td = `${mode === 'load' ? i * 55 : 0}ms`;
    const len = Math.round(n.len);
    nodes.push({
      cls: `gl-n gl-${kind}${vis ? '' : ' is-hidden'}${n.pulse && !off && vis ? ' is-pulse' : ''}`,
      d: n.d,
      w: w.toFixed(2),
      dash: !vis ? String(len) : off ? '2 6' : String(len),
      off: !vis ? String(len) : '0',
      shape,
      hx: n.x.toFixed(1),
      hy: n.y.toFixed(1),
      hr: (r * 2.6).toFixed(1),
      delay: n.delay,
      td,
    });
    n.tw.forEach((t) => {
      twigs.push({
        cls: `gl-t gl-${kind}${vis ? '' : ' is-hidden'}`,
        d: t.d,
        dash: !vis ? String(t.len) : off ? '1 5' : String(t.len),
        off: !vis ? String(t.len) : '0',
        x: t.x,
        y: t.y,
        r: t.r,
        td,
      });
    });
  });

  let items = [];
  if (mode === 'hero') {
    items = [
      { id: '[CLAIM]', sub: 'Nœud central', tx: CX + 12, ty: CY - 8, side: 'r', y: 120 },
      { id: '[CORPUS]', sub: ctx.corpusLabel || 'Corpus arXiv local', tx: geo.nodes[3].x, ty: geo.nodes[3].y, side: 'r', y: 540 },
    ];
  } else if (mode === 'load') {
    items = [{ id: '[CLAIM]', sub: 'Nœud central', tx: CX - 12, ty: CY - 10, side: 'l', y: 90 }];
    items.push({ id: `[AGENT_0${step + 1}]`, sub: ctx.agent, tx: CX + 30, ty: CY - 28, side: 'r', y: 90 });
  } else {
    items = [{ id: '[VERDICT]', sub: ctx.label, tx: CX - 12, ty: CY - 10, side: 'l', y: 90 }];
    if (ctx.conf != null) {
      const ang = -Math.PI / 2 + (ctx.conf / 100) * Math.PI * 2;
      items.push({ id: '[CONFIDENCE]', sub: `${fmtPct(ctx.conf)} %`, tx: CX + 40 * Math.cos(ang), ty: CY + 40 * Math.sin(ang), side: 'r', y: 600 });
    }
    ctx.evidence.forEach((e, i) => {
      const n = geo.nodes[i];
      if (!n) return;
      items.push({ id: `[EVIDENCE_0${i + 1}]`, sub: `${e.short} · ${fmtPct(e.nli)} %`, tx: n.x, ty: n.y, side: n.x < CX ? 'l' : 'r', y: n.y });
    });
  }

  const hasConf = mode === 'result' && ctx.conf != null;
  const ringDash = hasConf ? `${((ctx.conf / 100) * RING_LEN).toFixed(1)} ${RING_LEN}` : `0 ${RING_LEN}`;
  const centerCls = mode === 'load' ? 'gl-center gl-loading' : mode === 'result' ? `gl-center tone-${ctx.tone}` : 'gl-center';

  let aria = 'Graphe d’évidence décoratif : réseau de passages autour de l’affirmation.';
  if (mode === 'result') {
    aria = `Graphe d’évidence : ${ctx.kept} passages retenus (${ctx.counts[0]} soutiennent, ${ctx.counts[1]} contredisent, ${ctx.counts[2]} neutres) et ${ctx.disc} écartés autour de l’affirmation. Verdict : ${ctx.label}.`;
  }
  if (mode === 'load') aria = `Graphe d’évidence en construction, étape ${step + 1} sur 6.`;

  return { nodes, twigs, callouts: place(items), ringDash, centerCls, aria };
}
