# 🔬 Plateforme de Vérification de Claims Scientifiques par RAG et Agents IA

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20(Local)-black.svg)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Plateforme autonome et **100 % locale** de vérification d'affirmations scientifiques (*claims*) en informatique (sections `cs.*` d'arXiv), combinant **recherche d'information hybride**, **modèles d'inférence en langage naturel (NLI)** et un **pipeline multi-agents à boucle active**.

---

## 📌 Présentation du Projet

Face à la prolifération des publications scientifiques en informatique et aux risques critiques d'hallucinations des LLMs conventionnels (citations et DOI inventés), ce projet propose une alternative rigoureuse, auditable et sans dépendance au cloud :

* **Zéro hallucination de citations :** Chaque extrait cité est validé **mot pour mot** dans l'article source.
* **Honnêteté épistémique :** Le système sait répondre *"Preuves insuffisantes"* ou *"Sujet débattu"* au lieu de forcer une conclusion erronée.
* **100 % Local & Gratuit :** Aucune donnée n'est transmise à des API tierces (respect de la confidentialité, coût nul).

---

## 🏗️ Architecture du Système

Le système est piloté par un graphe d'états **LangGraph** articulé autour de **5 agents spécialisés** :

```text
                  ┌─────────────────────────────┐
                  │ 1. Agent Analyseur          │
                  │ Décomposition & requêtes    │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │ 2. Agent Récupérateur       │
                  │ BM25 + ChromaDB + Reranker  │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │ 3. Agent Évaluateur         │
                  │ Stance: Soutient / Contredit│
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │ 4. Agent Vérificateur       │
                  │ Contrôle exact des citations│
                  └──────────────┬──────────────┘
                                 │
             [Preuves insuffisantes ?] ──► (Boucle itérative vers Agent 1)
                                 │
                                 ▼ (Preuves validées)
                  ┌─────────────────────────────┐
                  │ 5. Agent Synthèse           │
                  │ Verdict, Confiance & Source │
                  └─────────────────────────────┘
```

### Les 5 Agents Spécialisés :
1. **Agent Analyseur :** Décompose l'affirmation et génère plusieurs sous-requêtes de recherche ciblées.
2. **Agent Récupérateur :** Recherche hybride (lexicale **BM25** + sémantique **ChromaDB**) suivie d'un reclassement par **Cross-Encoder**.
3. **Agent Évaluateur (Stance Detection) :** Détermine si le passage soutient, contredit ou est neutre via une double validation (**LLM + DeBERTa NLI**).
4. **Agent Vérificateur :** Contrôle textuel déterministe pour s'assurer que chaque extrait existe textuellement dans le papier arXiv source.
5. **Agent Synthèse :** Rédige le rapport final avec DOI, liens et calcule un score de confiance composite.

🔁 **Boucle de rétroaction active :** Si les preuves collectées sont jugées insuffisantes, l'orchestrateur reformule des requêtes plus précises et relance une recherche ciblée.

---

## 🛠️ Stack Technologique (100% Locale)

| Composant | Technologie | Justification |
| :--- | :--- | :--- |
| **LLM** | [Groq](https://console.groq.com/) via `init_chat_model` (LangChain) ; fournisseur et modèle réglables par variables d'environnement | Rédaction de la réponse et reformulation des requêtes. API distante : seuls le claim et les passages récupérés lui sont envoyés. Désactivable (`LLM_ENABLED=false`) |
| **Embeddings** | `bge-small-en-v1.5` (*sentence-transformers*) | Modèle d'embeddings dense compact et précis pour l'anglais scientifique |
| **Base Vectorielle** | [ChromaDB](https://www.trychroma.com/) (ou FAISS) | Base vectorielle embarquée et persistante en local |
| **Recherche Lexicale** | `rank_bm25` | Complémentaire au dense retrieval pour les acronymes et termes exacts |
| **Reranker** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Reclassement haute précision du Top-K passages |
| **Validation NLI** | `DeBERTa-v3-MNLI` | Détection de posture logique sans hallucination |
| **Orchestration Multi-Agents** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Modélisation en graphe cyclique avec boucle conditionnelle |
| **Backend & Interface** | FastAPI + Streamlit | API REST moderne et interface utilisateur interactive |
| **Conteneurisation** | Docker & Docker Compose | Déploiement et reproductibilité clés en main |

---

## 🚀 Installation & Démarrage

### Prérequis
* **Python 3.12** installé sur votre machine
* **Node.js** (pour construire l'interface React)
* Une **clé API Groq** (gratuite : https://console.groq.com/keys) pour la réponse rédigée. Sans clé, le pipeline fonctionne quand même, sans réponse rédigée.

### 1. Cloner le projet
```bash
git clone https://github.com/votre-compte/Projet-S5.git
cd Projet-S5
```

### 2. Créer et activer l'environnement virtuel Python (3.12)
```powershell
# Sous Windows PowerShell :
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances
```powershell
pip install -r requirements.txt
```

### 4. Configurer le LLM (Groq)
```powershell
copy .env.example .env
# puis éditez .env : GROQ_API_KEY=<votre clé>
```
Variables d'environnement (fichier `.env`, jamais commité) :

| Variable | Défaut | Rôle |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | (vide) | Clé API Groq. Absente : le LLM est désactivé, le reste fonctionne |
| `LLM_PROVIDER` | `groq` | Fournisseur passé à `init_chat_model` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Modèle ([liste Groq](https://console.groq.com/docs/models)) |
| `LLM_ENABLED` | `true` | `false` désactive toute génération |
| `LLM_TEMPERATURE`, `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_MAX_TOKENS` | `0`, `30`, `2`, `800` | Réglages d'appel |
| `LLM_ANSWER_LANGUAGE` | `French` | Langue de la réponse rédigée |

### 5. Construire le corpus, puis lancer l'application
```powershell
python scripts/run_collector.py --limit 100
python scripts/run_chunker.py
python scripts/run_indexer.py --reset
python scripts/run_hybrid_search.py --reindex      # construit l'index BM25
cd frontend; npm ci; npm run build; cd ..
python scripts/run_api.py                          # option : --port 8001 si le 8000 est occupé
```
Ouvrez ensuite http://127.0.0.1:8000. (Développement de l'interface : `npm run dev` dans `frontend/`, avec `VITE_API_TARGET` si l'API n'est pas sur le port 8000.)

**Interface** : design ALETHIA (thèmes sombre/clair, graphe d'évidence animé), polices auto-hébergées dans `frontend/src/assets/fonts/`. La progression des agents est lue en temps réel sur `/api/verify/stream` (SSE), boucle de rétroaction comprise. Le nom du produit se change dans `frontend/src/brand.js`.

### Docker (alternative aux étapes 2 à 5)
Une seule image contient l'API et l'interface. Le corpus (`data/`) est monté en volume et la clé Groq est lue dans `.env` au lancement (jamais copiée dans l'image) :
```powershell
docker compose up --build -d      # puis http://127.0.0.1:8001 (APP_PORT pour changer le port)
docker compose logs -f            # suivre le démarrage
docker compose down               # arrêter
```
Le premier démarrage télécharge les modèles Hugging Face (≈ 0,5 Go) dans le volume `hf_cache`. Si `data/` est vide, construisez le corpus dans le conteneur : `docker compose run --rm app python scripts/run_collector.py --limit 100`, puis `run_chunker.py`, `run_indexer.py --reset` et `run_hybrid_search.py --reindex`.

---

## 📊 Évaluation Scientifique

Le système est validé selon un protocole scientifique rigoureux :
* **Jeux de données :** Benchmark académique **SciFact** + un jeu de test de **~100 claims en informatique** annotés manuellement.
* **Métriques de Recherche (*Retrieval*) :** Recall@k, MRR (Mean Reciprocal Rank), nDCG.
* **Métriques de Classification :** Accuracy et Macro-F1 score par classe (*Soutient, Contredit, Neutre*).
* **Fidélité (*Faithfulness*) :** Taux de citations exactes mot pour mot vs taux d'hallucinations résiduelles.
* **Étude d'Ablation :** Mesure comparative progressive :
  $$\text{RAG Simple} \longrightarrow \text{RAG Hybride} \longrightarrow \text{Hybride + Reranker} \longrightarrow \text{Multi-Agents Complet}$$

---

## 👥 Auteurs & Encadrement

* **Étudiants :** Bilal MALIH • Oussama MAHI
* **Encadrante :** Pr. Karima MOUMANE
* **Cadre académique :** Projet de Semestre 5 (S5) — Cursus Ingénieur / Master Informatique
