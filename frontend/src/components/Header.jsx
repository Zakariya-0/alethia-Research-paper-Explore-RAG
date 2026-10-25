import Icon from './Icon.jsx';
import { BRAND } from '../brand.js';
import { llmPill } from '../lib/report.js';

export default function Header({ stats, theme, onToggleTheme }) {
  const pills = stats
    ? [
        `${stats.total_papers_indexed} articles`,
        `${stats.total_chunks} chunks`,
        'FAISS',
        'BM25',
        'Cross-Encoder',
        'DeBERTa-v3',
      ]
    : [];
  const llm = llmPill(stats);
  const dark = theme === 'dark';

  return (
    <header className="wrap hdr">
      <div className="brand">
        <svg viewBox="0 0 32 32" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" style={{ color: 'var(--accent-text)' }}>
          <path d="M16 16C12 12 9 9 5 7M16 16C20 11 23 8 27 5M16 16C17 21 16 25 13 29M16 16C22 18 26 21 29 25" />
          <circle cx="16" cy="16" r="3.2" fill="currentColor" stroke="none" />
          <circle cx="5" cy="7" r="1.8" fill="currentColor" stroke="none" />
          <circle cx="27" cy="5" r="1.8" fill="currentColor" stroke="none" />
          <circle cx="13" cy="29" r="1.8" fill="currentColor" stroke="none" />
          <circle cx="29" cy="25" r="1.8" fill="currentColor" stroke="none" />
        </svg>
        <div>
          <b>{BRAND.name}</b>
          <div className="cap">{BRAND.tagline}</div>
        </div>
      </div>
      <ul className="pills" aria-label="État du système">
        {pills.map((label) => (
          <li className="pill" key={label}>{label}</li>
        ))}
        {!stats && (
          <li className="pill pill-tone tone-warn"><span className="dot dot-warn" />API injoignable</li>
        )}
        {stats && (
          <li className={`pill ${llm.warn ? 'pill-tone tone-warn' : ''}`} title={stats.llm?.reason || ''}>
            <span className={`dot ${llm.warn ? 'dot-warn' : ''}`} />
            {llm.label}
          </li>
        )}
      </ul>
      <button className="btn theme" type="button" aria-label={dark ? 'Passer en mode clair' : 'Passer en mode sombre'} onClick={onToggleTheme}>
        <Icon name={dark ? 'sun' : 'moon'} />
      </button>
    </header>
  );
}
