# Analyse de Research-paper-Explorer

Analyse réalisée le 24 octobre 2026 sur la branche `analyse-claude`, à partir du commit `61744d6`.

**Convention de lecture.** Les `fichier:ligne` se rapportent au commit `61744d6`. En cours de session, les fichiers de `src/` et `scripts/` (29 fichiers) ont été reformatés par un outil extérieur à cette analyse, probablement un formateur de l'éditeur. Une comparaison d'AST avec HEAD montre que le code est identique : seule la mise en forme a changé, et les numéros de ligne du working tree ne correspondent donc plus. Aucun fichier du projet n'a été modifié par l'analyse ; ce fichier `ANALYSE.md` est le seul ajout.

Mention : **mesuré** = observé à l'exécution ; **lu** = constaté dans le code sans l'exécuter.

---

## 1. Résumé exécutif

Le projet est un prototype qui **fonctionne de bout en bout** une fois installé : collecte arXiv, chunking, index FAISS + BM25, fusion RRF, reranking, NLI DeBERTa, graphe LangGraph à 5 nœuds, API FastAPI et interface React. Il s'installe sans erreur avec les versions les plus récentes (aucune rupture d'API malgré des dépendances non épinglées). Sur un corpus de 100 articles, une vérification coûte 2 s à chaud (16,5 s à froid).

Ses **promesses centrales ne sont pas tenues** :
- Le vérificateur « mot pour mot » compare un extrait à lui-même : il répond toujours « valide », même sur un passage hors sujet (`nodes.py:342-343`). L'interface l'affiche en dur.
- Aucun garde-fou de pertinence : le claim d'exemple « Transformers outperform RNNs… » reçoit le verdict **RÉFUTÉ à 64,3 %**, fondé sur un article de méta-apprentissage par renforcement sans rapport (score reranker 0,17 %).
- Le README annonce un LLM local (Ollama), ChromaDB, Streamlit et Docker : **aucun n'existe dans le code**.
- Aucun test, aucune évaluation (SciFact promis), aucune version épinglée.

**Note globale : 4,5 / 10.** Structure modulaire propre, contrats Pydantic, boucle LangGraph bornée, exécution locale fluide : bonne base de prototype. Mais le cœur du produit (garantie anti-hallucination, verdict fiable) est faux ou non vérifié, et la documentation promet des composants absents.

---

## 2. Architecture réelle et schéma de flux

```
arXiv API ─► collector.py ─► data/raw/arxiv_papers_metadata.jsonl      (abstracts uniquement, pas de PDF)
                           ─► chunker.py  ─► data/processed/arxiv_chunks.jsonl   (800 car. / 150 overlap)
                           ├─► vector_store.py : FAISS IndexFlatIP + bge-small (384d)   data/vector_db/
                           └─► bm25_retriever.py : BM25Okapi (pickle)                    data/bm25/

POST /api/verify {claim, max_retries}                         (src/api/app.py)
 └─ graphe LangGraph (src/agents/graph.py)
      analyseur ─► récupérateur ─► évaluateur ─┬─ needs_feedback ─► analyseur   (max_retries tours)
                                               └─ sinon ─► vérificateur ─► synthèse
   analyseur    : regex de mots-clés + règles de catégories (aucun LLM)
   récupérateur : FAISS + BM25 ─► RRF (12 candidats) ─► cross-encoder MiniLM ─► top 3
   évaluateur   : DeBERTa-v3-small NLI par passage ─► verdict agrégé
   vérificateur : audit « mot pour mot » (tautologique, voir C1)
   synthèse     : score composite (consensus 35 %, reranker 25 %, NLI 25 %, récence 15 %)
 └─► JSON ─► frontend React (frontend/src/App.jsx, un seul composant de 510 lignes)
```

**Points d'entrée :** `scripts/run_*.py` (CLI), `src.api.app:app` via `scripts/run_api.py`, et `npm run dev` / `npm run build`.
**Ordre réel des scripts :** `run_collector` → `run_chunker` → `run_indexer` (FAISS) → `run_hybrid_search` (construit BM25) → `run_api`. `run_reranker`, `run_verification` et `run_agentic_pipeline` sont des démos indépendantes. `scale_up_corpus.py` enchaîne collecte, chunking, FAISS et BM25, mais **vide les index existants**.
**Dépendances externes :** API arXiv ; modèles HuggingFace `BAAI/bge-small-en-v1.5` (134 Mo), `cross-encoder/ms-marco-MiniLM-L-6-v2` (91 Mo), `cross-encoder/nli-deberta-v3-small` (568 Mo). **Ollama n'est jamais appelé** ; il est installé sur la machine d'analyse, sans effet.
**Configuration :** tout est dans `src/config.py` (aucune variable d'environnement). Device `cpu` codé en dur à trois endroits.

### Incohérences documentation / code

| Documentation | Réalité | Preuve |
|---|---|---|
| `streamlit run app.py` | Aucun `app.py`, aucun `import streamlit` ; l'interface est en React | `README.md:118-120` |
| Dossier `Projet-S5` | Le dépôt s'appelle Research-paper-Explorer | `README.md:95-96`, `config.py:2`, `config.py:13` |
| LLM local Ollama (Llama 3.1 / Qwen) | Aucune référence dans `src/` ni `scripts/` ; aucun LLM dans le pipeline | `README.md:63`, `README.md:75` |
| ChromaDB | FAISS est utilisé ; `chromadb` déclaré mais jamais importé | `vector_store.py:29`, `requirements.txt:8` |
| `VECTOR_DB_DIR` | Défini deux fois (`chroma_db` puis `vector_db`) | `config.py:23`, `config.py:56` |
| Docker & Docker Compose | Aucun Dockerfile | `README.md:83` |
| Évaluation SciFact, Recall@k, ablation | Aucun code d'évaluation | `README.md:125-133` |
| Licence MIT | Aucun fichier `LICENSE` | `README.md:7` |
| « DeBERTa-v3-MNLI » | Modèle réel : `nli-deberta-v3-small` | `config.py:97` |
| Corpus de 2 445 articles (message de commit) | `data/` est ignoré par git et absent du dépôt | `.gitignore:31` |

---

## 3. Résultat de l'exécution

**Machine :** Windows 10, Python 3.12.8, Node 24.15.0, 7,9 Go de RAM, CPU seul (Intel HD 5500), 7,2 Go libres. Installation « réduite » (seulement ce que le code importe), `faiss-cpu` ajouté à part car absent de `requirements.txt`.

### Ce qui marche (mesuré)

| Étape | Résultat | Durée |
|---|---|---|
| `pip install` (torch 2.14.1, sentence-transformers 6.1.0, transformers 5.20.2, langgraph 1.2.16, faiss-cpu 1.15.1…) | OK, venv de 1,2 Go | 469 s |
| `npm ci` | OK, 25 paquets | 36 s |
| `run_collector.py --limit 100` | 100 articles | 14 s |
| `run_chunker.py` | 254 chunks (83 mots en moyenne) | 5 s |
| `run_indexer.py` | OK (inclut le téléchargement de bge, 134 Mo) | 380 s |
| `run_hybrid_search.py --compare` (construit BM25) | OK | 31 s |
| `run_agentic_pipeline.py` (1ʳᵉ exécution, inclut le téléchargement de MiniLM + DeBERTa, ≈ 660 Mo) | OK | 932 s |
| `npm run build` | OK, bundle JS de 240 ko | 5 s |
| Démarrage de l'API | `/api/health` OK | 25 s |
| `POST /api/verify`, 1ʳᵉ requête (chargement des modèles) | HTTP 200 | **16,5 s** |
| `POST /api/verify`, requêtes suivantes | HTTP 200 | **1,8 s** (1 passe) à **6,6 s** (3 passes) |
| `POST /api/verify/stream` | HTTP 200, `text/event-stream` | 7 s |
| Interface servie par FastAPI, claim lancé depuis le navigateur | Affichage complet | ≈ 5 s |

Aucun échec à l'installation. Les modèles HuggingFace affichent seulement un avertissement sur les liens symboliques (Windows sans mode développeur) : sans conséquence.

### Ce qui échoue ou pose problème (mesuré)

1. **Port 8000 occupé** sur cette machine par `wslrelay.exe` et Docker Desktop : `run_api.py` plante au démarrage (`[Errno 10048]`). Contournement : `--port 8001`. Mais `frontend/vite.config.js:11` pointe le proxy en dur sur `127.0.0.1:8000`, donc `npm run dev` ne joint plus l'API sans modifier ce fichier. L'interface a été testée via `npm run build`, que l'API sert elle-même à `/`.
2. **Le claim d'exemple donne un faux verdict.**
3. **Le flux SSE perd des données** : le message `DONE` contient 1 log sur 12 et 1 entrée de rétroaction sur 2.
4. **Un claim composé d'espaces** (`"     "`) est accepté (HTTP 200). `{"claim":"abc"}` et `max_retries: 9` donnent bien 422.

### Requête et réponse réelles : le claim demandé

Requête :

```
POST http://127.0.0.1:8001/api/verify
{"claim":"Transformers outperform RNNs on long-range dependency tasks","max_retries":2}
```

Réponse (extrait) :

```json
{
  "verdict": "REFUTE",
  "confidence_score": 0.6427,
  "supports_count": 0, "refutes_count": 1, "neutral_count": 2,
  "confidence_breakdown": {
    "score_consensus": 0.7, "score_reranker": 0.0017,
    "score_nli": 0.9893, "score_recency": 1.0, "composite_score": 0.6427
  },
  "evidences": [
    {"paper_id": "2610.31656", "title": "METTA: Memory-Enhanced Test-Time Adaptation for Meta-Reinforcement Learning",
     "nli_stance": {"label": "REFUTES", "confidence": 0.989}, "rerank_score": 0.0017,
     "citation_check": {"is_valid": true, "match_type": "EXACT"}}
  ],
  "iteration_count": 1
}
```

Le corpus de 100 articles récents ne contient aucun article sur le sujet. La récupération renvoie donc des passages hors sujet (score reranker ≈ 0,002). DeBERTa juge pourtant l'un d'eux « contradictoire » à 98,9 %, le système le traite comme une preuve, l'audit le déclare « EXACT », et la confiance globale atteint 64,3 %. Le message du récupérateur indique bien « pertinence faible (0,1 %) », mais rien n'en tient compte.

### Quatre claims, mesurés

| Claim | Verdict | Confiance | Passes | Durée | Lecture |
|---|---|---|---|---|---|
| Transformers outperform RNNs on long-range dependency tasks | RÉFUTÉ | 64,3 % | 1 | 2,0 s | **Faux** : source hors sujet |
| Bananas are made of pure bitcoin and cure all diseases | PREUVES INSUFFISANTES | 24,6 % | 3 | 6,6 s | Correct, mais confiance non nulle |
| Reinforcement learning agents can adapt to unseen tasks using memory | SOUTENU | 84,7 % | 1 | 1,8 s | Plausible (article METTA pertinent, reranker 0,9998) |
| Diffusion models improve image synthesis | PREUVES INSUFFISANTES | 40,5 % | 3 | 5,0 s | Passages pertinents (reranker 0,86) jugés « neutres » |

Dans l'interface, le premier exemple (« Controverse Démontrée », claim *Video world models…*) donne **« Preuves insuffisantes », 33,9 %**, et non la controverse annoncée : les exemples dépendent du corpus de l'auteur, que cette machine n'a pas.

**Résolu :** le doute sur une double sigmoïde du reranker est levé. Les `rerank_score` couvrent bien 0,0000 à 0,9998, ils ne sont pas comprimés.

---

## 4. Constats classés par gravité

### Critique

| Gravité | Fichier:ligne | Problème | Correctif |
|---|---|---|---|
| Critique | `src/agents/nodes.py:342-343` (copié dans `scripts/run_verification.py:129-133`) | L'audit « mot pour mot » vérifie `raw_text[:120]` contre `raw_text` : **toujours valide** (mesuré : EXACT sur tous les passages hors sujet). `citation_audit_passed` est écrit (`nodes.py:384`) mais jamais lu. L'UI affiche « 100 % MOT POUR MOT » en dur (`App.jsx:429`). | Extraire la phrase qui justifie le verdict, la vérifier contre l'abstract original du JSONL ; afficher le vrai `citation_check`. |
| Critique | `src/agents/nodes.py:224-234`, `253-321` | **Aucun garde-fou de pertinence** : `retrieval_status` (`LOW_RELEVANCE`) est écrit mais jamais lu ; le top 3 va toujours au NLI. Résultat mesuré : verdict RÉFUTÉ à 64,3 % sur une source hors sujet. | Seuil minimal sur `rerank_score` ; en dessous, `PREUVES_INSUFFISANTES` sans appeler le NLI. |
| Critique | `requirements.txt` | `faiss` est importé (`vector_store.py:29`) mais absent des dépendances : tout échoue au premier import sur une machine neuve. | Ajouter `faiss-cpu`. |

### Important

| Gravité | Fichier:ligne | Problème | Correctif |
|---|---|---|---|
| Important | `citation_verifier.py:89`, `:119-127` ; `chunker.py:104-105` | « Mot pour mot » est faux de trois façons : comparaison insensible à la casse, match flou ≥ 0,90 déclaré valide, texte cité nettoyé (`$` et commandes LaTeX supprimés). | Mode strict sensible à la casse ; conserver le texte brut à côté du texte nettoyé. |
| Important | `confidence_scorer.py:224-251` | Le score mélange pertinence et vérité. Mesuré : 24,6 %, 33,9 % et 40,5 % de « confiance » sur des verdicts « preuves insuffisantes ». | Mettre le score à 0 (ou le masquer) pour `PREUVES_INSUFFISANTES`. |
| Important | `nodes.py:174-177` | La reformulation est déterministe : le 2ᵉ réessai répète la même requête et la même recherche. Mesuré : 3 passes pour un résultat identique (6,6 s au lieu de 2 s). | Faire dépendre la reformulation de la requête précédente ; arrêter si la recherche ne change pas. |
| Important | `nodes.py:125-138`, `:210` | Catégories détectées par sous-chaîne (`"git"` dans `"digital"`) ; si une seule ressort, filtre dur sur `primary_category` (mesuré : le claim d'exemple est filtré sur `cs.LG` seul). | Mots entiers, filtre doux (bonus) plutôt qu'exclusion. |
| Important | `src/api/app.py:185` | Dans `/verify/stream`, `accumulated_state.update(...)` écrase les champs à reducer. **Mesuré :** `DONE.agent_logs` = 1 au lieu de 12, `feedback_history` = 1 au lieu de 2. | Concaténer ces champs à la main ou utiliser `stream_mode="values"`. |
| Important | `vector_store.py:185-186` vs `bm25_retriever.py:170-171` | FAISS ajoute sans dédupliquer, BM25 reconstruit tout : relancer `run_indexer` sans `--reset` double les vecteurs et fausse le RRF (lu). | Index idempotent : vider ou dédupliquer par `chunk_id`. |
| Important | `vector_store.py:211-213`, `app.py:89-97` | Sans index, l'API répond 200 « preuves insuffisantes » ; `/health` répond toujours « healthy » (lu). | HTTP 503 si les index sont vides ; `/health` qui teste les index. |
| Important | `nli_evaluator.py:198-216` | Un seul passage « refute » parmi cinq « supports » donne `SUJET_DEBATTU` ; conditions `n_sup >= n_ref` / `n_ref > n_sup` mortes ; verdict par passage, consensus par article. | Aligner les unités, pondérer par la confiance. |
| Important | `bm25_retriever.py:80` | Pas de stemming : « RNNs » et « Transformers » ne correspondent pas à « RNN » et « transformer » ; les tokens `llama-3.1` ne sont pas éclatés. | Lemmatisation légère, sous-tokens. |
| Important | `collector.py:254-259`, `frontend/src/App.jsx:23-37` | Le corpus est constitué des N derniers abstracts par catégorie : couverture thématique faible ; les exemples de l'UI dépendent d'un snapshot (mesuré : l'exemple « controverse » donne autre chose). | Collecter par mots-clés ; supprimer ou rendre configurables les exemples. |
| Important | `README.md:125-133` | Évaluation promise (SciFact, Recall@k, ablation) absente ; seuils (NLI 0,60, reranker 0,25) non calibrés. | Script d'évaluation sur SciFact. |
| Important | (dépôt) | Aucun test, Dockerfile, `pyproject` ni CI. | Voir les 5 actions ci-dessous. |
| Important | `vite.config.js:11` | Proxy en dur sur `127.0.0.1:8000`, port pris par WSL/Docker sur la machine d'analyse (mesuré). | Variable d'environnement pour la cible du proxy et le port de l'API. |

### Mineur

| Gravité | Fichier:ligne | Problème | Correctif |
|---|---|---|---|
| Mineur | `hybrid_search.py:145`, `reranker.py` | Plusieurs chunks du même article occupent le top 3 (mesuré : METTA ×2, MRI diffusion ×2) : peu de diversité, consensus « article unique ». | Dédupliquer par `paper_id` avant le top-k. |
| Mineur | `chunker.py:142` | Le séparateur `". "` laisse des chunks commençant par `". "` (mesuré, visible dans les extraits affichés). | `keep_separator="end"` ou nettoyage du début de chunk. |
| Mineur | `app.py:70-76` | Un claim d'espaces passe la validation (mesuré). | `strip_whitespace` ou validateur Pydantic. |
| Mineur | `app.py:57-58` | CORS `*` avec `allow_credentials=True`, inutile derrière le proxy Vite ; n'importe quel site peut appeler l'API locale. | Origines explicites ou aucun CORS. |
| Mineur | `app.py:160`, `:217` | `str(e)` renvoyé au client. | Message générique, détail dans les logs. |
| Mineur | `bm25_retriever.py:124` | `pickle.load` exécuterait du code si le fichier était altéré. | Stocker les tokens en JSON. |
| Mineur | `App.jsx:76-92` | Progression des agents simulée par `setTimeout` (non annulés si le `fetch` échoue réseau) ; l'endpoint SSE existe mais n'est pas utilisé. | Consommer `/api/verify/stream`. |
| Mineur | `App.jsx:96` | Une erreur 422 affiche `[object Object]` (`detail` est un tableau). | Formater `errData.detail`. |
| Mineur | `App.jsx:324-365`, `app.py:120` | Poids « (35 %) » et dimension 384 codés en dur. | Lire `breakdown.weights` et `EMBEDDING_DIMENSION`. |
| Mineur | `nodes.py:362-363` | Valeurs par défaut inventées dans les preuves (`"2024-01-01"`, `"cs.AI"`, `"inconnu"`). | Rejeter l'élément ou exposer `null`. |
| Mineur | `nodes.py:47-83` | Singleton de modèles non thread-safe ; chargement à la première requête (mesuré : +14 s). | `lifespan` FastAPI qui préchauffe, verrou. |
| Mineur | `nodes.py:216-222` vs `config.py:89` | Candidats codés à 12 dans le nœud, 15 dans la config. | Lire la config. |
| Mineur | `nli_evaluator.py:157-163` | NLI appelé passage par passage. | Un seul `predict` par lot. |
| Mineur | `collector.py:183`, `:75` | `split("v")` casse les anciens identifiants arXiv ; `datetime.utcnow()` déprécié. | Regex `v\d+$` ; `datetime.now(timezone.utc)`. |
| Mineur | 23 fichiers | Bloc `sys.stdout.reconfigure` copié partout ; `logging.basicConfig` à l'import (`collector.py:46`) ; `sys.path.append` dans chaque script. | Un `pyproject.toml`, `pip install -e .`, `python -m`. |
| Mineur | `graph.py:117` | `verify_claim_agentic` n'impose aucun plafond (la limite de récursion de LangGraph est de 25) ; l'API plafonne à 3. | Valider `max_feedback_iterations` dans la fonction. |

### Dépendances

| Statut | Paquets |
|---|---|
| Manquant | `faiss-cpu` |
| Importé, non déclaré | `numpy`, `langchain-text-splitters` (transitifs) |
| Déclaré, jamais importé | `streamlit`, `chromadb`, `python-dotenv`, `requests`, `pypdf`, `langchain-community` ; `langchain` seulement pour son splitter |
| Versions | Toutes en `>=`. Les versions récentes ont fonctionné sans rupture lors de cette analyse, mais rien ne le garantit à l'avenir. `frontend/package-lock.json` existe. |

### Sécurité

Aucun secret dans le dépôt. Pas de LLM, donc pas d'injection de prompt aujourd'hui ; ce sera à traiter si Ollama est ajouté, car le corpus arXiv est un texte non fiable. Voir aussi CORS, fuite d'erreurs et `pickle` ci-dessus.

### Tests : les 5 à écrire en priorité

1. **`CitationVerifier` et nœud vérificateur** : exact, casse, citation altérée sous 0,90, et un test de non-régression qui échoue tant que C1 existe.
2. **`aggregate_verdict` et `route_evaluator_decision`**, en table : support seul, refute seul, mixte, tout neutre, vide, seuil à 0,60.
3. **Terminaison de la boucle** : `ModelRegistry` remplacé par des stubs ; le graphe s'arrête après `max_retries + 1` passes pour 0 à 3.
4. **RRF et indexation idempotente** : retrievers factices, `chunk_id` en double, filtre de catégorie, FAISS indexé deux fois.
5. **API avec `TestClient`** : 422 sur claim trop court ou vide, 503 sans index, `agent_logs` complet dans le flux SSE.

---

## 5. Plan d'amélioration priorisé

| # | Action | Effort |
|---|---|---|
| 1 | **Rendre le verdict honnête** : vraie vérification de citation (phrase de preuve contre l'abstract brut), seuil de pertinence avant le NLI, confiance nulle sans preuve concluante, affichage du vrai audit dans l'UI. Corrige C1, la pertinence et la confiance. | M |
| 2 | **Assainir les dépendances** : ajouter `faiss-cpu`, retirer les inutilisées (`streamlit`, `chromadb`, `pypdf`, `langchain-community`, `python-dotenv`), épingler les versions (fichier de lock), ajouter un `pyproject.toml`. | S |
| 3 | **Améliorer la récupération** : dédupliquer par article, stemming BM25, filtre de catégorie doux, FAISS idempotent, reformulation réellement différente à chaque tour. | M |
| 4 | **Ajouter les 5 tests** ci-dessus et une CI minimale (pytest + lint). | M |
| 5 | **Aligner doc et config** : corriger le README (Streamlit, Ollama, ChromaDB, Docker, nom du projet, licence), ports configurables (API et proxy Vite), corriger le flux SSE et le message d'erreur 422. Décider ensuite si un LLM local est réellement voulu. | S (doc et config) / L (LLM) |

---

## 6. Commandes pour relancer le projet sur votre machine

Toutes les commandes se lancent depuis `C:\Users\zakariya\Research-paper-Explorer` dans PowerShell. Le venv, les paquets, les données et les modèles sont **déjà installés** : si vous ne désinstallez rien, sautez directement à l'étape 6.

**1. Venv et paquets Python** (≈ 8 min, ≈ 250 Mo ; `faiss-cpu` manque dans `requirements.txt`, d'où la liste explicite) :

```bash
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install langgraph langchain-text-splitters pydantic rank-bm25 sentence-transformers torch arxiv fastapi uvicorn faiss-cpu numpy
```

**2. Frontend** (≈ 40 s) :

```bash
cd frontend; npm ci; npm run build; cd ..
```

**3. Données : 100 articles** (≈ 15 s ; relancez avec `--limit 3000` pour un corpus plus large, en comptant environ 3 s par page de 100 articles) :

```bash
.\venv\Scripts\python.exe scripts\run_collector.py --limit 100
```

**4. Chunking** (≈ 5 s) :

```bash
.\venv\Scripts\python.exe scripts\run_chunker.py
```

**5. Index FAISS puis BM25** (le premier appel télécharge bge-small, 134 Mo ; ne pas relancer `run_indexer.py` sans `--reset`) :

```bash
.\venv\Scripts\python.exe scripts\run_indexer.py --reset
.\venv\Scripts\python.exe scripts\run_hybrid_search.py --reindex
```

**6. Démarrer l'API et l'interface** (le port 8000 est pris par WSL/Docker sur cette machine, d'où `--port 8001`) :

```bash
.\venv\Scripts\python.exe scripts\run_api.py --port 8001
```

Ouvrez ensuite **http://127.0.0.1:8001**. La première vérification charge DeBERTa et le reranker (≈ 16 s, plus ≈ 660 Mo à télécharger la toute première fois) ; les suivantes durent 2 à 7 s.

**Mode développement** (rechargement à chaud) : `npm run dev` dans `frontend/` suppose l'API sur le port 8000. Si ce port est occupé, modifiez la cible du proxy dans `frontend/vite.config.js:11` pour la faire pointer sur le bon port.

**Test en ligne de commande, sans interface :**

```bash
.\venv\Scripts\python.exe scripts\run_agentic_pipeline.py --claim "Reinforcement learning agents can adapt to unseen tasks using memory" --verbose
```

**Arrêter l'API :** `Ctrl+C` dans son terminal.

---

## 7. Mise à jour du 25 octobre 2026 : LLM Groq ajouté

Suite à cette analyse, les modifications suivantes ont été faites sur la branche `analyse-claude` (non commitées). Les constats ci-dessus se rapportent au commit `61744d6` et les numéros de ligne ne sont plus valables pour les fichiers modifiés.

| Changement | Fichiers |
|---|---|
| **LLM Groq** créé avec `init_chat_model` ; fournisseur, modèle et réglages par variables d'environnement (`LLM_PROVIDER`, `LLM_MODEL`, `GROQ_API_KEY`…) ; sans clé, le pipeline fonctionne comme avant | `src/generation/` (nouveau), `src/config.py`, `.env.example`, `requirements.txt` |
| **Agent Générateur** (6ᵉ nœud) : réponse rédigée à partir des seuls passages retenus, au format JSON avec citations | `src/agents/nodes.py`, `graph.py`, `state.py` |
| **Reformulation des requêtes par le LLM** dans l'Analyseur (corrige le constat I3 : la 2ᵉ relance n'est plus identique) | `src/agents/nodes.py` |
| **Audit réel des citations** (constat C1) : phrase citée recherchée dans l'abstract stocké, casse comprise ; les citations du LLM sont auditées de la même façon | `src/agents/nodes.py`, `schemas.py` |
| **Seuil de pertinence** (constat I2) : passages sous 0,05 écartés ; la pertinence se juge contre le claim d'origine, pas la requête reformulée | `src/agents/nodes.py`, `config.py` |
| API : `answer` et `llm` dans `/api/verify`, statut dans `/api/stats` et `/api/health` | `src/api/app.py` |
| Interface : section « Réponse rédigée », citations ✓/✗, alerte si le LLM contredit DeBERTa, 6ᵉ étape « Générateur », bandeau et pastille LLM dynamiques, vrai badge d'audit, messages d'erreur 422 lisibles, proxy Vite configurable (`VITE_API_TARGET`) | `frontend/src/App.jsx`, `App.css`, `vite.config.js` |
| `faiss-cpu` ajouté aux dépendances (constat C2) ; README mis à jour (Groq, variables, commandes réelles) | `requirements.txt`, `README.md` |

**Limites connues.** La réponse du LLM n'a été testée qu'avec un **faux LLM** (aucune clé Groq disponible lors de l'analyse) : citation exacte vérifiée, citation altérée et `chunk_id` inventé rejetés, échec ou JSON invalide du LLM sans casser le pipeline, LLM désactivé. Un appel réel à Groq reste à valider. Avec Groq, le claim et les passages récupérés quittent la machine : le pipeline n'est plus « 100 % local » (le bandeau de l'interface l'indique). Non corrigés : le flux SSE perd des logs (I5), la confiance reste non nulle sans preuve (I9), aucun test automatisé.
