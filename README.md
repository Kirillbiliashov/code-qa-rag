# Code QA RAG

Ask natural-language questions about any Python codebase and get answers grounded in the actual source — not in whatever the LLM happened to memorize during training.

---

## The problem

Reading an unfamiliar codebase is slow. Generic chat models hallucinate APIs, miss project-specific patterns, and have no idea what lives inside *your* repo. Dumping the whole repository into a prompt does not scale — most codebases blow past any context window, and even when they fit, the model drowns in irrelevant tokens.

**Code QA RAG** solves this with a Retrieval-Augmented Generation pipeline tailored to source code:

1. You upload a repository as a `.zip` archive.
2. The system parses every `.py` file with Python's AST, splits it into **semantic chunks** (modules, classes, functions, methods), and indexes them in a vector database.
3. When you ask a question, only the chunks that are actually relevant are pulled back and handed to the LLM as grounding context.

The LLM answers from your code, not from its training data.

---

## How it works

### High-level RAG pipeline

```
            ┌──────────────────────────────────────────────────────────────┐
            │                       INGESTION PHASE                        │
            └──────────────────────────────────────────────────────────────┘

   repo.zip ──► unzip ──► AST parse ──► SemanticExtractor ──► FileChunks
                                                                  │
                                                                  ▼
                                              ┌───────────────────────────┐
                                              │  Dense embedding (ST)     │
                                              │  Sparse embedding (BM25)  │
                                              └─────────────┬─────────────┘
                                                            │
                                  ┌─────────────────────────┴─────────────┐
                                  ▼                                       ▼
                          ┌───────────────┐                       ┌──────────────┐
                          │   Qdrant      │                       │   MongoDB    │
                          │ (vectors +    │                       │ (raw chunk   │
                          │  metadata)    │                       │  source)     │
                          └───────────────┘                       └──────────────┘


            ┌──────────────────────────────────────────────────────────────┐
            │                      QUESTION-ANSWER PHASE                   │
            └──────────────────────────────────────────────────────────────┘

   user question
        │
        ▼
   ┌─────────────────────────┐   classify scope     ┌────────────────────────┐
   │  Retriever              │ ───────────────────► │  DeepSeek LLM          │
   │  (scope filter +        │   (module/class/     │  CLASSIFY_USER_INPUT   │
   │   hybrid search)        │    function)         └────────────────────────┘
   └────────────┬────────────┘
                │ dense + sparse query, RRF fusion
                ▼
        ┌───────────────┐         top-K chunk IDs
        │    Qdrant     │ ───────────────────────────┐
        └───────────────┘                            │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  MongoDB lookup      │
                                          │  (full source code)  │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                            ┌──────────────────────────────────────────┐
                            │   Map: per-chunk answer (QA_PROMPT)      │
                            │   N parallel LLM calls                   │
                            └────────────────────┬─────────────────────┘
                                                 │
                                                 ▼
                            ┌──────────────────────────────────────────┐
                            │   Reduce: merge answers (REDUCE_PROMPT)  │
                            └────────────────────┬─────────────────────┘
                                                 │
                                                 ▼
                                          final answer
```

### Ingestion in detail

When a user uploads a `.zip`, [services/ingestion_service.py](services/ingestion_service.py) extracts it, walks the tree with [RepositoryScanner](ingestion/repo_scanner.py) (which respects `.gitignore`), and parses every Python file with [SemanticExtractor](ingestion/extractor.py).

Instead of splitting code by line count, the extractor walks the AST and emits **semantic chunks**:

- **ModuleChunk** — file-level overview: imports, top-level constants, class signatures, function signatures.
- **ClassChunk** — a single class, its docstring, attributes, and method signatures.
- **FunctionChunk / MethodChunk** — a single function or method with its full body.

This matters because retrieval can now be **scoped**. A question like *"how does the QAService work?"* is class-level; *"how is the answer reduced?"* is function-level. Chunking by AST node lets us match the granularity of the question.

Each chunk is embedded twice:

- **Dense** vector via `flax-sentence-embeddings/st-codesearch-distilroberta-base` — captures semantic similarity ("classify input" matches "categorize question").
- **Sparse** BM25 vector via `Qdrant/bm25` — captures exact-token matches ("QAService", "reduce", specific identifiers).

Both vectors land in Qdrant under the same point, along with metadata (`type`, `repo_id`, file path, line range). The raw source code is stored separately in MongoDB and joined back in at answer time, keeping the vector store lean.

### Retrieval in detail

The [Retriever](retrieval/retriever.py) does three things:

1. **Classifies the question** with the `CLASSIFY_USER_INPUT_PROMPT` — is it about a `module`, a `class`, or a `function`? This becomes a Qdrant filter on the `type` field, so we only search chunks of the right granularity.
2. **Runs hybrid search**: dense and sparse queries are issued as Qdrant `Prefetch` calls and fused with **Reciprocal Rank Fusion (RRF)**. Dense alone misses literal identifier matches; sparse alone misses paraphrases. RRF gets both.
3. **Filters by `repo_id`** so users only see results from their own upload.

### Answer generation in detail

[QAService.answer](services/qa_service.py) takes the top-K chunk IDs from Qdrant, fetches their full source from MongoDB, then runs a **map-reduce** over the LLM:

- **Map step** — for each retrieved chunk, run `QA_PROMPT` ([qa/prompts.py](qa/prompts.py)) in parallel via `asyncio.gather`. Each call gets *one* chunk as context and is told to refuse (return `""`) if the question is unrelated. This isolates noise: a chunk that does not help cannot poison the others.
- **Reduce step** — concatenate the non-empty partial answers and feed them to `REDUCE_PROMPT`, which deduplicates, drops irrelevant content, and produces one clean final answer.

#### The prompts

| Prompt | File | Job |
|---|---|---|
| `CLASSIFY_USER_INPUT_PROMPT` | [retrieval/retriever.py](retrieval/retriever.py) | Decide the scope of the question (`module` / `class` / `function`) so retrieval can filter by chunk type. |
| `QA_PROMPT` | [qa/prompts.py](qa/prompts.py) | Per-chunk answer. Strict: answer only from the given context, return `""` if the question is unrelated. |
| `REDUCE_PROMPT` | [qa/prompts.py](qa/prompts.py) | Merge per-chunk answers into one coherent, deduplicated response. |

#### Why map-reduce instead of one big prompt

Stuffing all K chunks into a single prompt works, but it has costs: the model gets distracted by irrelevant chunks, attention degrades over long context, and there is no graceful failure mode for a chunk that does not contribute. Map-reduce treats each chunk independently, so weak retrievals self-eliminate (`""`) and strong ones get full attention.

---

## Tech stack

| Layer | What |
|---|---|
| Web framework | **FastAPI** + Uvicorn |
| LLM | **DeepSeek** via `llama-index-llms-deepseek` |
| Prompt templating | `llama-index-core` (`ChatPromptTemplate`) |
| Dense embeddings | `sentence-transformers` — `flax-sentence-embeddings/st-codesearch-distilroberta-base` (768-dim) |
| Sparse embeddings | `fastembed` — `Qdrant/bm25` |
| Vector store | **Qdrant** (hybrid dense + sparse, RRF fusion) |
| Document store | **MongoDB** (raw chunk source + repo metadata + per-IP query quotas) |
| Code parsing | Python `ast`, `pathspec` for `.gitignore` |
| Frontend | Static HTML + CSS (see [static/](static/)) |

See [requirements.txt](requirements.txt) for exact pinned versions.

---

## What it looks like

### 1. Upload a repository

![Upload screen](images/upload-screen.png)

### 2. Ask a question

![QA input](images/qa-input.png)

### 3. Get an answer grounded in your code

![QA output](images/qa-output.png)

---

## Running it locally

### Prerequisites

- Python 3.11+
- A running **Qdrant** instance (local Docker is easiest)
- A **MongoDB** connection string (Atlas free tier or local)
- A **DeepSeek** API key

### 1. Clone and set up a virtualenv

```bash
git clone <this-repo>
cd code-qa-rag

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Qdrant locally

The simplest path is Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

Then point [config/settings.py](config/settings.py) at it. By default `QDRANT_URL` is hardcoded to `http://192.168.106.2:6333` — change it to `http://localhost:6333` for a typical local setup.

### 3. Provide a `.env`

Create a `.env` file in the project root:

```bash
DEEPSEEK_API_KEY=sk-...your-key...
MONGO_DB_CONN_STRING=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true
```

Both keys are required — DeepSeek powers question classification, per-chunk answers, and the reduce step; MongoDB stores repo metadata, chunk source, and rate-limit state.

### 4. Start the server

```bash
uvicorn app:app --reload
```

Open <http://localhost:8000>, upload a `.zip` of any Python repo, and start asking questions.

---

## Current limitation: not hosted publicly

This project is **not currently deployed to a public URL**. The reason is purely operational: the embedding models are loaded locally and weigh **500 MB+** combined, which pushes the deployment footprint past the free tier of every hosting platform we evaluated.

The plan to lift this restriction:

- Swap local sentence-transformers for a hosted embeddings API (e.g. DeepSeek or another provider with affordable embedding endpoints), so the running container stays small.
- Or move the model-heavy ingestion path to a separate background worker so the QA service itself stays slim.

Once that is done, a public hosted version will follow.
