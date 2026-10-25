import Icon from './Icon.jsx';
import { EXAMPLES, MAX_CLAIM, MIN_CLAIM } from '../lib/examples.js';

export default function ClaimForm({ claim, onClaim, tries, onTries, onSubmit, llm }) {
  const ready = claim.trim().length >= MIN_CLAIM;

  return (
    <>
      <h1 className="display">Vérifiez une affirmation <em>scientifique.</em></h1>

      <div className="palette" role="img" aria-label="Légende des couleurs : soutient, contredit, neutre, écarté">
        <i style={{ background: 'var(--g-sup)' }} />
        <i style={{ background: 'var(--g-con)' }} />
        <i style={{ background: 'var(--g-neu)' }} />
        <i style={{ background: 'var(--g-off)' }} />
      </div>
      <div className="palette-l">
        <span className="cap">Soutient</span>
        <span className="cap">Contredit</span>
        <span className="cap">Neutre</span>
        <span className="cap">Écarté</span>
      </div>

      <form
        className="card xh form"
        aria-label="Formulaire de vérification"
        onSubmit={(e) => {
          e.preventDefault();
          if (ready) onSubmit();
        }}
      >
        <label className="cap" htmlFor="claim">[AFFIRMATION]</label>
        <textarea
          id="claim"
          className="ta"
          rows={3}
          lang="en"
          maxLength={MAX_CLAIM}
          value={claim}
          onChange={(e) => onClaim(e.target.value)}
          placeholder="Saisissez une affirmation scientifique (en anglais, comme le corpus)…"
        />
        {claim.trim().length > 0 && !ready && (
          <p className="hint" role="status">Au moins {MIN_CLAIM} caractères sont nécessaires.</p>
        )}

        <div className="chips" role="group" aria-label="Exemples d’affirmations">
          <span className="cap">Exemples</span>
          {EXAMPLES.map((ex) => (
            <button key={ex.label} className="chip" type="button" title={ex.claim} onClick={() => onClaim(ex.claim)}>
              {ex.label}
            </button>
          ))}
        </div>

        <div className="form-row">
          <div className="field">
            <label className="cap" htmlFor="tries">Max essais</label>
            <select id="tries" className="sel" value={tries} onChange={(e) => onTries(Number(e.target.value))}>
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={3}>3</option>
            </select>
          </div>
          <button className="btn btn-primary" type="submit" disabled={!ready}>
            Vérifier formellement
            <Icon name="arrow" />
          </button>
        </div>

        {llm?.enabled && (
          <p className="notice">
            <Icon name="globe" />
            <span>LLM actif : l’affirmation et les passages sélectionnés sont envoyés à une API distante ({llm.provider}).</span>
          </p>
        )}
      </form>
    </>
  );
}
