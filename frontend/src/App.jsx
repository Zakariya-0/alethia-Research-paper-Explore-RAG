import { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';
import { ApiError, fetchStats, verifyStream } from './lib/api.js';
import { AGENTS, buildReport } from './lib/report.js';
import { BRAND } from './brand.js';
import Header from './components/Header.jsx';
import Icon from './components/Icon.jsx';
import ClaimForm from './components/ClaimForm.jsx';
import { EXAMPLES } from './lib/examples.js';
import AgentStepper from './components/AgentStepper.jsx';
import EvidenceGraph, { GraphLegend } from './components/EvidenceGraph.jsx';
import VerdictPanel from './components/VerdictPanel.jsx';
import LlmAnswer from './components/LlmAnswer.jsx';
import EvidenceCards from './components/EvidenceCards.jsx';
import TechnicalDetails from './components/TechnicalDetails.jsx';

const THEME_KEY = 'axon-theme';

function initialTheme() {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'dark' || saved === 'light') return saved;
  } catch {
    /* stockage indisponible : thème par défaut */
  }
  return 'dark';
}

const NODE_INDEX = Object.fromEntries(AGENTS.map((a, i) => [a.node, i]));

export default function App() {
  const [theme, setTheme] = useState(initialTheme);
  const [stats, setStats] = useState(null);
  const [screen, setScreen] = useState('idle'); // idle | loading | result
  const [claim, setClaim] = useState(EXAMPLES[0].claim);
  const [tries, setTries] = useState(3); // nombre d'essais max (rétroactions = essais - 1)
  const [submitted, setSubmitted] = useState('');
  const [step, setStep] = useState(0);
  const [retry, setRetry] = useState(0);
  const [feedback, setFeedback] = useState('');
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  useEffect(() => {
    fetchStats().then(setStats);
  }, []);

  useEffect(() => {
    document.title = BRAND.documentTitle;
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* préférence non mémorisée */
    }
  }, [theme]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const submit = useCallback(async (text = claim) => {
    const value = text.trim();
    if (value.length < 5) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setSubmitted(value);
    setError(null);
    setReport(null);
    setStep(0);
    setRetry(0);
    setFeedback('');
    setScreen('loading');
    const started = performance.now();
    let attempts = 0;

    try {
      const data = await verifyStream({
        claim: value,
        maxRetries: tries - 1,
        signal: controller.signal,
        onEvent: (ev) => {
          if (ev.type !== 'NODE_UPDATE') return;
          const idx = NODE_INDEX[ev.node];
          if (ev.node === 'evaluateur' && ev.needs_feedback) {
            // Preuves insuffisantes : le graphe reboucle vers l'Analyseur.
            attempts += 1;
            setRetry(attempts);
            setFeedback(ev.feedback_reason || '');
            setStep(0);
          } else if (idx != null) {
            setStep(Math.min(idx + 1, AGENTS.length - 1));
          }
        },
      });
      setReport(buildReport(data, tries, (performance.now() - started) / 1000));
      setScreen('result');
    } catch (err) {
      if (err.name === 'AbortError') return;
      setError({
        title: err instanceof ApiError && err.kind === 'network' ? 'API injoignable' : 'Erreur serveur',
        message: err.message || 'Le pipeline n’a pas répondu.',
      });
      setScreen('idle');
    }
  }, [claim, tries]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setScreen('idle');
  }, []);

  const reset = useCallback(() => {
    setReport(null);
    setError(null);
    setScreen('idle');
    window.scrollTo({ top: 0 });
  }, []);

  const isIdle = screen === 'idle';
  const isLoading = screen === 'loading';
  const isResult = screen === 'result' && report;
  const llm = stats?.llm;
  const corpusLabel = stats ? `${stats.total_papers_indexed} articles · ${stats.total_chunks} chunks` : 'Corpus arXiv local';

  return (
    <div className="nx" data-theme={theme} data-glow="normal">
      <Header stats={stats} theme={theme} onToggleTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')} />

      <main className="wrap">
        {error && isIdle && (
          <div className="banner banner-err" role="alert" style={{ marginBottom: 8 }}>
            <Icon name="server" />
            <div><b>{error.title}</b><span>{error.message} Votre affirmation est conservée.</span></div>
            <button className="btn btn-sm" type="button" onClick={() => submit()}>
              <Icon name="refresh" />Réessayer
            </button>
          </div>
        )}

        {isIdle && llm && !llm.enabled && (
          <div className="banner banner-warn" role="status" style={{ marginBottom: 8 }}>
            <Icon name="key" />
            <div>
              <b>{/clé api/i.test(llm.reason || '') ? 'Clé API absente' : 'LLM désactivé'}</b>
              <span>
                Le verdict reste calculé par la recherche, le reranker et le NLI ; aucune réponse rédigée ni audit de
                citations n’est produit. {llm.reason ? `${llm.reason}.` : ''}
              </span>
            </div>
          </div>
        )}

        {isResult && (
          <section className="res-head" aria-label="Affirmation vérifiée">
            <div>
              <span className="cap">[AFFIRMATION]</span>
              <h2 className="h2" lang="en">{report.claim}</h2>
              <p className="meta">{report.metaLine}</p>
            </div>
            <button className="btn btn-sm" type="button" onClick={reset}>
              <Icon name="refresh" />Nouvelle vérification
            </button>
          </section>
        )}

        <div className={`row ${isIdle ? 'row-idle' : ''}`}>
          <div className="col-l">
            {isIdle && (
              <ClaimForm claim={claim} onClaim={setClaim} tries={tries} onTries={setTries} onSubmit={() => submit()} llm={llm} />
            )}
            {isLoading && (
              <AgentStepper
                claim={submitted}
                step={step}
                retry={retry}
                tries={tries}
                llm={llm}
                lastFeedback={feedback}
                onCancel={cancel}
              />
            )}
            {isResult && <VerdictPanel report={report} />}
          </div>

          <div className="col-r">
            <div className={isIdle ? 'stage-bg' : 'stage-panel card'}>
              {isIdle && <EvidenceGraph mode="hero" corpusLabel={corpusLabel} />}
              {isLoading && <EvidenceGraph mode="load" step={step} agent={AGENTS[Math.min(step, 5)].name} />}
              {isResult && <EvidenceGraph mode="result" report={report} />}
              {!isIdle && <GraphLegend />}
            </div>
          </div>
        </div>

        {isIdle && (
          <>
            <ol className="pipe" aria-label="Les six agents">
              {AGENTS.map((a, i) => (
                <li key={a.node}>
                  <div className="pipe-n">0{i + 1}<small>/06</small></div>
                  <b>{a.name}</b>
                  <span>
                    {a.node === 'recuperateur' && stats
                      ? `Recherche hybride BM25 + FAISS dans ${stats.total_chunks} chunks.`
                      : a.desc}
                  </span>
                </li>
              ))}
            </ol>
            <div className="empty">
              <p>Aucun résultat pour l’instant. Le verdict, le graphe d’évidence et les passages cités apparaîtront ici.</p>
              <span className="cap">[RÉSULTATS] 00 / 00</span>
            </div>
          </>
        )}

        {isResult && (
          <>
            <LlmAnswer llm={report.llm} onRetry={() => submit(report.claim)} />
            <EvidenceCards report={report} />
            <TechnicalDetails report={report} />
          </>
        )}
      </main>
    </div>
  );
}
