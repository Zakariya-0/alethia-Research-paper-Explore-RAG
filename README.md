# ALETHIA : vérification de claims scientifiques par RAG et agents IA

Application web de vérification d'affirmations scientifiques (*claims*) en informatique. Elle interroge un corpus local d'abstracts arXiv (catégories `cs.*`) grâce à une recherche hybride, juge chaque passage par inférence en langage naturel (NLI), fait rédiger une réponse sourcée par un LLM, puis contrôle chaque citation avant de rendre un verdict chiffré.

Projet de Semestre 5 (S5), cursus ingénieur / master informatique.

## Sommaire

1. [Présentation](#1-présentation)
2. [Fonctionnalités](#2-fonctionnalités)
3. [Architecture](#3-architecture)
4. [Stack technique](#4-stack-technique)
5. [Structure du dépôt](#5-structure-du-dépôt)
6. [Installation et exécution](#6-installation-et-exécution)
7. [Configuration](#7-configuration)
8. [Utilisation](#8-utilisation)
9. [Référence de l'API](#9-référence-de-lapi)
10. [Limites connues](#10-limites-connues)
11. [Évaluation et perspectives](#11-évaluation-et-perspectives)
12. [Auteurs et encadrement](#12-auteurs-et-encadrement)

---

## 1. Présentation

### Contexte et problème

Le volume de publications en informatique croît plus vite que la capacité de les lire. Les assistants à base de LLM répondent vite, mais peuvent inventer des citations ou des références, sans que l'utilisateur puisse le détecter. Pour une affirmation scientifique, il faut au contraire savoir **sur quels textes** repose la réponse, et **dans quelle mesure** ils la soutiennent ou la contredisent.

### Approche

ALETHIA ne demande pas à un LLM de « savoir » : il lui demande de **rédiger à partir de passages récupérés**, puis vérifie son travail.

1. Les passages pertinents sont retrouvés dans le corpus (BM25 et vecteurs, fusionnés, puis reclassés).
2. Un modèle NLI dédié décide, passage par passage, s'il soutient, contredit ou ne tranche pas.
3. Le LLM rédige une réponse courte avec des citations exactes.
4. Chaque citation est recherchée telle quelle dans l'abstract stocké. Une citation introuvable est signalée, jamais masquée.
5. Un score de confiance composite et un verdict à quatre états sont calculés, avec le détail de chaque dimension.

Le système sait répondre **« Preuves insuffisantes »** ou **« Sujet débattu »** plutôt que de forcer une conclusion.

### Ce que le projet n'est pas

- Ce n'est pas un détecteur de vérité : le score est un indice pondéré, pas une probabilité que l'affirmation soit vraie.
- Il ne lit que les **abstracts** du corpus indexé, pas les articles complets.
- Il ne remplace pas la lecture des articles sources : l'interface fournit les liens arXiv et PDF.

---

## 2. Fonctionnalités

- Recherche hybride (BM25 + FAISS, fusion RRF) suivie d'un reclassement par Cross-Encoder et d'un seuil de pertinence.
- Détection de posture par NLI (DeBERTa-v3) : soutient, contredit, neutre.
- Réponse rédigée par un LLM (Groq par défaut, fournisseur interchangeable) avec citations auditées.
- Boucle de rétroaction : si les preuves sont insuffisantes, l'Analyseur reformule la recherche, dans la limite d'un nombre d'essais choisi.
- Score de confiance décomposé en quatre dimensions, verdict à quatre états.
- Interface web : suivi en direct des agents, graphe d'évidence, cartes de preuves, journal d'audit, JSON brut, thèmes clair et sombre.
- Fonctionnement dégradé propre : sans clé API, le verdict est calculé sans réponse rédigée.
- Déploiement en un seul conteneur Docker.

---

## 3. Architecture

### Vue d'ensemble

```text
Navigateur (React + Vite)
        |
        |  HTTP / SSE
        v
API FastAPI  (sert aussi l'interface construite : frontend/dist)
        |
        v
Graphe LangGraph : 6 agents
        |
        +--> Recherche hybride : BM25 + FAISS + RRF --> data/ (chunks, index)
        +--> Reclassement      : Cross-Encoder
        +--> Posture NLI       : DeBERTa-v3
        +--> LLM (facultatif)  : init_chat_model, Groq par défaut
                                 reçoit l'affirmation et les passages retenus
```

L'API FastAPI sert à la fois l'interface construite (`frontend/dist`) et les endpoints `/api/*`. Il n'y a ni base de données externe ni service tiers obligatoire : le corpus et les index sont des fichiers dans `data/`. Seul le LLM est distant, et il est facultatif.

### Pipeline multi-agents

```text
Affirmation
    |
    v
[1 Analyseur] --> [2 Récupérateur] --> [3 Évaluateur]
      ^                                       |
      |   preuves insuffisantes               |
      +-- et essais restants -----------------+
                                              |
                                              | sinon
                                              v
[6 Synthèse] <-- [5 Vérificateur] <-- [4 Générateur]
      |
      v
Rapport + verdict
```

| # | Agent | Rôle |
|---|---|---|
| 1 | **Analyseur** | Extrait les mots-clés techniques et cible les catégories arXiv. Aux tours suivants, reformule la requête (par le LLM s'il est actif, sinon par une heuristique). |
| 2 | **Récupérateur** | Recherche hybride sur 15 candidats, reclassement Cross-Encoder, conservation des 3 meilleurs passages dont le score dépasse le seuil de pertinence. |
| 3 | **Évaluateur** | Calcule la posture NLI de chaque passage et le verdict provisoire. Déclenche la rétroaction si le verdict est « Preuves insuffisantes » et qu'il reste des essais. |
| 4 | **Générateur** | Fait rédiger au LLM une réponse fondée uniquement sur les passages, avec des citations exactes. Ne fait jamais échouer le pipeline : sans clé, sans passage ou en cas d'erreur, l'étape est signalée et ignorée. |
| 5 | **Vérificateur** | Recherche chaque citation dans l'abstract stocké (correspondance exacte, sans appariement approximatif) et compare la position du LLM au verdict NLI. |
| 6 | **Synthèse** | Calcule le score de confiance composite et assemble le rapport (`ClaimVerificationReport`, modèle Pydantic). |

L'état partagé entre agents (`VerificationState`) contient notamment l'historique de rétroaction et le journal d'audit, cumulés d'un nœud à l'autre.

### Recherche d'information

1. **Collecte** : abstracts arXiv des catégories `cs.AI`, `cs.CL`, `cs.LG`, `cs.CV`, `cs.RO`, `cs.SE`, `cs.CR`.
2. **Découpage** : chunks de 800 caractères avec 150 de chevauchement.
3. **Recherche dense** : embeddings `BAAI/bge-small-en-v1.5` (384 dimensions) dans un index FAISS `IndexFlatIP`.
4. **Recherche lexicale** : BM25Okapi (`k1 = 1,5`, `b = 0,75`), utile pour les acronymes et les termes exacts.
5. **Fusion** : Reciprocal Rank Fusion avec `k = 60`.
6. **Reclassement** : Cross-Encoder `ms-marco-MiniLM-L-6-v2`. Les passages dont le score est inférieur à 0,05 sont écartés avant le NLI : un passage hors sujet ne doit jamais servir de preuve. Quand la requête a été reformulée, la pertinence est toujours jugée contre l'affirmation d'origine.

### Verdict et score de confiance

**Verdict** à partir des postures NLI conclusives (probabilité d'au moins 0,60) :

| Verdict | Condition |
|---|---|
| Soutenu | Au moins un passage soutient de façon conclusive, aucun ne contredit |
| Réfuté | Au moins un passage contredit de façon conclusive, aucun ne soutient |
| Sujet débattu | Au moins un passage soutient et au moins un contredit |
| Preuves insuffisantes | Aucun passage conclusif, ou aucun passage retenu |

**Score de confiance** : somme pondérée de quatre dimensions.

| Dimension | Poids | Signification |
|---|---|---|
| Consensus | 35 % | Accord entre les passages retenus |
| Cross-Encoder | 25 % | Pertinence sémantique moyenne des passages |
| DeBERTa | 25 % | Certitude logique des postures NLI |
| Récence | 15 % | Moyenne de `exp(-0,10 × âge en années)` |

### Audit des citations

- Chaque citation produite par le LLM est cherchée telle quelle, casse comprise, dans l'abstract de l'article cité, brut puis nettoyé comme au découpage.
- La phrase retenue comme preuve pour chaque passage est auditée de la même façon.
- Les citations non retrouvées sont conservées et marquées « Non retrouvée ».
- Si la position du LLM diffère du verdict NLI, une alerte s'affiche : **le verdict affiché reste celui du NLI**.

L'audit prouve qu'une citation existe dans l'abstract. Il ne prouve pas que la phrase de réponse qui l'entoure soit exacte.

---

## 4. Stack technique

| Couche | Technologie | Rôle |
|---|---|---|
| Orchestration | LangGraph | Graphe d'états avec arête conditionnelle (boucle de rétroaction) |
| LLM | LangChain `init_chat_model` + `langchain-groq` | Fournisseur et modèle choisis par variables d'environnement |
| Embeddings | sentence-transformers, `bge-small-en-v1.5` | Représentation dense des chunks |
| Index vectoriel | FAISS (`faiss-cpu`) | Recherche par similarité |
| Recherche lexicale | `rank-bm25` | BM25Okapi |
| Reclassement | Cross-Encoder `ms-marco-MiniLM-L-6-v2` | Pertinence fine des candidats |
| NLI | Cross-Encoder `nli-deberta-v3-small` | Soutient, contredit, neutre |
| Découpage | `langchain-text-splitters` | Chunks avec chevauchement |
| Collecte | bibliothèque `arxiv` | Récupération des métadonnées et abstracts |
| API | FastAPI, Uvicorn, Pydantic | REST, streaming SSE, validation des données |
| Interface | React 19, Vite 8 | Application monopage, polices auto-hébergées |
| Déploiement | Docker, Docker Compose | Une image : API et interface |

Les modèles tournent sur CPU. Aucune carte graphique n'est nécessaire.

---

## 5. Structure du dépôt

```text
.
├── src/
│   ├── config.py              Chemins, hyperparamètres, variables d'environnement
│   ├── data/                  Collecte arXiv et découpage en chunks
│   ├── retrieval/             FAISS, BM25, fusion hybride, Cross-Encoder
│   ├── verification/          NLI, audit des citations, score de confiance, schémas
│   ├── generation/            Création du LLM (init_chat_model) et génération de la réponse
│   ├── agents/                Graphe LangGraph, nœuds, état partagé
│   └── api/app.py             API FastAPI + service de l'interface
├── scripts/                   Lignes de commande (collecte, index, pipeline, API)
├── frontend/                  Interface React (Vite)
├── data/                      Corpus et index (non versionné)
├── Dockerfile                 Image unique (interface + API)
├── docker-compose.yml
├── requirements.txt           Dépendances complètes
├── requirements.docker.txt    Sous-ensemble d'exécution
└── .env.example               Modèle de configuration
```

---

## 6. Installation et exécution

### Prérequis

- Python 3.12
- Node.js 20.19 ou plus (22.12 ou plus recommandé), uniquement pour construire l'interface hors Docker
- Une clé API Groq (gratuite, https://console.groq.com/keys). Elle est facultative : sans elle, le verdict est calculé sans réponse rédigée.
- Environ 2 Go de disque pour les dépendances et 0,5 Go pour les modèles, téléchargés au premier usage

### Configurer la clé

```bash
cp .env.example .env      # Windows : copy .env.example .env
```

Renseignez ensuite `GROQ_API_KEY` dans `.env`. Le fichier est ignoré par Git : ne le commitez jamais.

### Option A : Docker (recommandée)

Une seule image contient l'API et l'interface. Le corpus (`data/`) est monté depuis votre machine, et la clé est lue dans `.env` au lancement sans être copiée dans l'image.

```bash
docker compose up --build -d
```

Ouvrez http://127.0.0.1:8001 (variable `APP_PORT` pour changer le port).

```bash
docker compose logs -f     # suivre le démarrage
docker compose down        # arrêter
```

- Le premier build prend plusieurs minutes (PyTorch en version CPU).
- Au premier lancement, les modèles Hugging Face se téléchargent dans le volume `hf_cache` ; la première vérification est donc lente.
- Si `data/` est vide, construisez le corpus dans le conteneur (voir ci-dessous), en remplaçant `python` par `docker compose run --rm app python`.

### Option B : exécution locale

**1. Environnement virtuel et dépendances**

```bash
python -m venv venv
venv\Scripts\activate              # Linux / macOS : source venv/bin/activate
pip install -r requirements.txt
```

**2. Construire le corpus et les index** (une seule fois)

```bash
python scripts/run_collector.py --limit 100
python scripts/run_chunker.py
python scripts/run_indexer.py --reset
python scripts/run_hybrid_search.py --reindex
```

Pour un corpus plus grand : `python scripts/scale_up_corpus.py --target 3000` (collecte, découpage et indexation automatisés, plus long).

**3. Construire l'interface**

```bash
cd frontend
npm ci
npm run build
cd ..
```

**4. Lancer l'application**

```bash
python scripts/run_api.py
```

Ouvrez http://127.0.0.1:8000. Si le port 8000 est occupé, utilisez `--port 8001`.

La documentation interactive de l'API est disponible sur `/docs`.

### Développement de l'interface

```bash
cd frontend
npm run dev
```

Le serveur Vite (port 5173) redirige `/api` vers l'API. Si l'API n'écoute pas sur le port 8000, définissez la variable `VITE_API_TARGET` (par exemple `http://127.0.0.1:8001`). La commande `npm run lint` lance le linter.

### Utilisation en ligne de commande

Le pipeline complet s'exécute sans interface :

```bash
python scripts/run_agentic_pipeline.py --claim "Diffusion models improve image synthesis" --verbose
python scripts/run_agentic_pipeline.py --claim "Diffusion models improve image synthesis" --json
```

---

## 7. Configuration

### Variables d'environnement (`.env`)

| Variable | Défaut | Rôle |
|---|---|---|
| `GROQ_API_KEY` | (vide) | Clé du fournisseur. Absente : le LLM est désactivé, le reste fonctionne |
| `LLM_PROVIDER` | `groq` | Fournisseur passé à `init_chat_model` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Modèle (liste : https://console.groq.com/docs/models) |
| `LLM_ENABLED` | `true` | `false` désactive toute génération |
| `LLM_TEMPERATURE` | `0` | Température d'échantillonnage |
| `LLM_TIMEOUT_SECONDS` | `30` | Délai maximal d'un appel |
| `LLM_MAX_RETRIES` | `2` | Nouvelles tentatives réseau |
| `LLM_MAX_TOKENS` | `800` | Longueur maximale de la sortie |
| `LLM_ANSWER_LANGUAGE` | `French` | Langue de la réponse rédigée |

Pour utiliser un autre fournisseur, installez le paquet `langchain-<fournisseur>` correspondant et renseignez `LLM_PROVIDER`, `LLM_MODEL` et la variable de clé propre au fournisseur.

### Hyperparamètres (`src/config.py`)

| Paramètre | Valeur | Rôle |
|---|---|---|
| `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP` | 800 / 150 | Découpage |
| `RRF_K` | 60 | Fusion hybride |
| `DEFAULT_RERANK_CANDIDATES` / `DEFAULT_RERANK_TOP_K` | 15 / 3 | Candidats puis passages conservés |
| `RERANK_MIN_RELEVANCE` | 0,05 | Seuil de pertinence du Cross-Encoder |
| `NLI_CONFIDENCE_THRESHOLD` | 0,60 | Probabilité minimale pour qu'une posture soit conclusive |
| `WEIGHT_*` | 0,35 / 0,25 / 0,25 / 0,15 | Poids du score de confiance |
| `RECENCY_DECAY_RATE` | 0,10 | Décroissance de la récence, par an |

---

## 8. Utilisation

1. Saisissez une affirmation **en anglais** (le corpus et les modèles sont anglophones), de 5 à 500 caractères.
2. Choisissez le nombre maximal d'essais : 1 désactive la boucle de rétroaction, 3 autorise deux reformulations.
3. Lancez la vérification et suivez les six agents en direct.
4. Lisez le résultat :
   - le **verdict** et la **confiance**, avec la décomposition en quatre dimensions ;
   - le **graphe d'évidence** : un nœud par passage, couleur selon la posture, trait épais selon la pertinence, pointillés pour les passages écartés ;
   - la **réponse du LLM**, ses citations (« Vérifiée » ou « Non retrouvée ») et les alertes éventuelles ;
   - les **passages** classés, avec leurs scores, le changement de rang et les liens arXiv et PDF ;
   - le **journal d'audit** des agents et la **réponse JSON** brute de l'API.

Exemples à essayer :

```text
Memory-enhanced meta-reinforcement learning agents can adapt quickly to unseen tasks at test time without retraining.
Diffusion models improve image synthesis.
Photosynthetic bacteria can transmit encrypted data between colonies.
```

La première est adossée à un article du corpus, la deuxième est générale, la troisième est hors corpus et doit aboutir à « Preuves insuffisantes ».

---

## 9. Référence de l'API

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/api/health` | État du service et mode d'exécution (LLM actif ou non) |
| `GET` | `/api/stats` | Taille du corpus, état des index, état du LLM |
| `POST` | `/api/verify` | Exécute le pipeline et renvoie le rapport complet |
| `POST` | `/api/verify/stream` | Même pipeline en Server-Sent Events : événements `START`, `NODE_UPDATE` (un par agent), puis `DONE` avec le rapport |

Corps de requête de `/api/verify` et `/api/verify/stream` :

```json
{ "claim": "Diffusion models improve image synthesis", "max_retries": 2 }
```

`claim` fait de 5 à 500 caractères ; `max_retries` est le nombre maximal de rétroactions, de 0 à 3.

Principaux champs de la réponse : `verdict`, `verdict_fr`, `confidence_score`, `confidence_breakdown`, `supports_count`, `refutes_count`, `neutral_count`, `discarded_passages`, `evidences[]`, `answer` (texte, position, citations et leur statut d'audit), `llm` (état, modèle, latence), `agent_logs[]`, `feedback_history[]`, `iteration_count`.

---

## 10. Limites connues

- **Corpus restreint** : la configuration par défaut indexe 100 articles. Une affirmation sur un sujet absent donne « Preuves insuffisantes », ce qui est le comportement voulu et non une erreur.
- **Abstracts seulement** : les articles complets ne sont ni lus ni indexés.
- **Anglais** : les affirmations doivent être en anglais.
- **Trois passages par vérification** : le pipeline conserve les 3 meilleurs passages après reclassement.
- **NLI sur affirmations simples** : le modèle juge une proposition à la fois. Les affirmations longues ou composées donnent souvent un résultat neutre.
- **Seuils non calibrés** : les seuils de pertinence et de décision ont été fixés sur quelques dizaines de cas observés, sans benchmark.
- **LLM distant et variable** : avec Groq, l'affirmation et les passages sélectionnés sont envoyés à une API externe. Une réponse du modèle peut être vide ou invalide ; un seul nouvel essai est tenté, puis l'étape est signalée en erreur sans bloquer le verdict.
- **Audit partiel** : seules les citations sont contrôlées, pas la prose de la réponse rédigée.
- **Démarrage à froid** : la première requête après un lancement charge les modèles et prend de 10 secondes à quelques minutes, selon la machine et le téléchargement initial.
- **Pas de tests automatisés** pour le moment.

---

## 11. Évaluation et perspectives

L'évaluation quantitative n'est pas encore réalisée. Le protocole envisagé :

- jeux de données : benchmark **SciFact** et un jeu d'environ 100 affirmations en informatique annotées manuellement ;
- recherche : Recall@k, MRR, nDCG ;
- classification : exactitude et F1 macro par classe (soutient, contredit, neutre) ;
- fidélité : taux de citations retrouvées dans les abstracts ;
- ablation : RAG simple, puis RAG hybride, puis hybride avec reclassement, puis pipeline multi-agents complet.

Pistes d'amélioration : calibrer les seuils sur un benchmark, indexer les articles complets, élargir le corpus, ajouter des tests automatisés, auditer la prose de la réponse rédigée.

---

## 12. Auteurs et encadrement

- **Étudiants** : Bilal MALIH, Oussama MAHI
- **Encadrante** : Pr. Karima MOUMANE
- **Cadre académique** : Projet de Semestre 5 (S5), cursus ingénieur / master informatique
