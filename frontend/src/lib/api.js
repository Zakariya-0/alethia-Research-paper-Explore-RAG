// Client de l'API FastAPI (URLs relatives : proxy Vite en dev, même origine en production).

export class ApiError extends Error {
  constructor(message, { kind = 'server', status } = {}) {
    super(message);
    this.kind = kind; // 'server' (réponse d'erreur) ou 'network' (API injoignable)
    this.status = status;
  }
}

async function errorFromResponse(response) {
  const body = await response.json().catch(() => ({}));
  // Les erreurs de validation FastAPI (422) renvoient une liste d'objets
  const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join(' ; ') : body.detail;
  return new ApiError(detail || `Erreur serveur HTTP ${response.status}`, { status: response.status });
}

export async function fetchStats() {
  try {
    const res = await fetch('/api/stats');
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

/**
 * Lance la vérification en flux SSE et appelle onEvent pour chaque événement
 * (START, NODE_UPDATE, ...). Renvoie le rapport final (événement DONE).
 */
export async function verifyStream({ claim, maxRetries, signal, onEvent }) {
  let response;
  try {
    response = await fetch('/api/verify/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ claim, max_retries: maxRetries }),
      signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new ApiError('Le serveur ne répond pas. Vérifiez que l’API est démarrée.', { kind: 'network' });
  }
  if (!response.ok) throw await errorFromResponse(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let done = null;

  const handle = (block) => {
    const line = block.split('\n').find((l) => l.startsWith('data:'));
    if (!line) return;
    const event = JSON.parse(line.slice(5).trim());
    if (event.type === 'ERROR') throw new ApiError(event.error || 'Erreur du pipeline.');
    if (event.type === 'DONE') done = event;
    onEvent?.(event);
  };

  try {
    for (;;) {
      const { value, done: finished } = await reader.read();
      if (finished) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop();
      blocks.forEach(handle);
    }
    if (buffer.trim()) handle(buffer);
  } catch (err) {
    if (err instanceof ApiError || err.name === 'AbortError') throw err;
    throw new ApiError('La connexion au serveur a été interrompue.', { kind: 'network' });
  }

  if (!done) throw new ApiError('Le flux s’est terminé sans rapport final.');
  return done;
}
