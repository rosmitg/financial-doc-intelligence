# Financial Document Intelligence System

A production-oriented financial document intelligence system demonstrating permission-aware RAG, hybrid retrieval, prompt optimisation, automated evaluation, and LLM observability.

Built as an AI engineering portfolio project focused on retrieval quality, grounded generation, access-controlled retrieval, evaluation, and production deployment patterns.

---

## Architecture

```text
Documents
   ↓
LlamaIndex ingestion + metadata enrichment
   ↓
HuggingFace embeddings (all-MiniLM-L6-v2, local)
   ↓
PostgreSQL + pgvector (Neon)
   ↓
Role-filtered hybrid retrieval
(dense vector + BM25 lexical)
   ↓
LangChain LCEL chain
   ↓
Claude Haiku
   ↓
Answer + Sources + Latency

Evaluation (offline)        → RAGAS
Observability (per request) → LangSmith
Prompt optimisation         → DSPy
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Document indexing | LlamaIndex |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | pgvector on Neon |
| Retrieval | Hybrid search — dense vector + BM25 lexical |
| LLM | Claude Haiku via Anthropic API |
| Retrieval chain | LangChain LCEL |
| Prompt optimisation | DSPy BootstrapFewShot |
| Evaluation | RAGAS with LLM-as-judge |
| Observability | LangSmith |
| API | FastAPI |
| Containerisation | Docker |
| CI | GitHub Actions |

---

## Evaluation Results

Evaluated on a **10-question labelled golden dataset** spanning multiple access roles.

Results shown below are from the latest local evaluation run.

| Metric | Score | Threshold | Status |
|---|---:|---:|---|
| Faithfulness | 0.955 | 0.70 | ✅ PASS |
| Answer Relevancy | 0.796 | 0.70 | ✅ PASS |
| Context Precision | 0.900 | 0.50 | ✅ PASS |
| Context Recall | 0.900 | 0.50 | ✅ PASS |
| Avg Latency | 3.22s | — | — |

---

## Key Design Decisions

- **Authorisation before retrieval:** classification filters are applied as SQL `WHERE` clauses before retrieved chunks reach the LLM. Restricted content is not retrieved first and filtered afterward.

- **Local embeddings:** HuggingFace MiniLM keeps embedding cost at zero and avoids dependency on an external embedding API.

- **Hybrid retrieval:** dense semantic search is combined with BM25 lexical matching to improve retrieval of exact financial terms, acronyms, figures, and policy language.

- **Evaluation as a quality gate:** RAGAS evaluation runs against a fixed golden dataset, allowing retrieval and prompt changes to be checked for regressions.

- **Neon + pgvector:** relational metadata and vector retrieval are kept in the same database, reducing architectural complexity.

- **DSPy prompt optimisation:** prompts are optimised against evaluation criteria rather than relying only on manual prompt tuning.

---

## Role-Aware Retrieval

The system demonstrates role-aware retrieval by applying SQL-level classification filters before vector search results are passed to the LLM.

| Role | Allowed Access |
|---|---|
| `public` | PUBLIC |
| `junior_analyst` | PUBLIC + INTERNAL |
| `senior_analyst` | PUBLIC + INTERNAL + CONFIDENTIAL |
| `executive` | PUBLIC + INTERNAL + CONFIDENTIAL + RESTRICTED |

### Important Security Note

This project currently accepts a `role` value directly in the API request for demonstration and evaluation purposes.

In a production system, the client would **not** be trusted to choose its own role.

Instead:

```text
Authenticated User
      ↓
JWT / Session
      ↓
Backend derives role
      ↓
Authorisation policy
      ↓
SQL-level retrieval filter
```

The retrieval enforcement itself is designed around the correct security principle:

> Authorisation should be applied before or during retrieval, not after restricted content has already reached the LLM.

The main missing production component is an authentication layer that binds a verified identity to its allowed role.

---

## Features

### RAG Pipeline

- LlamaIndex-based document ingestion
- Metadata enrichment during indexing
- Local MiniLM embeddings
- PostgreSQL + pgvector storage
- HNSW vector indexing
- Dense semantic retrieval
- BM25 lexical retrieval
- Hybrid retrieval fusion
- Source metadata returned with answers

### Access-Controlled Retrieval

- Four classification levels:
  - PUBLIC
  - INTERNAL
  - CONFIDENTIAL
  - RESTRICTED
- Role-based access mapping
- SQL-level retrieval filtering
- Restricted chunks are excluded before generation

### Prompt Optimisation

- DSPy BootstrapFewShot
- Golden evaluation dataset
- Optimised prompt artefacts saved under `storage/`

### Evaluation

- RAGAS evaluation suite
- Claude used as LLM judge
- Metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall
- Latency tracking
- Quality thresholds for regression checking

### Observability

- LangSmith tracing
- Request-level visibility into chain execution
- Useful for latency analysis, prompt debugging, and retrieval inspection

---

## Project Structure

```text
financial-doc-intelligence/
├── app/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── chains/
│   │   ├── retrieval_chain.py
│   │   └── dspy_optimizer.py
│   │
│   ├── evaluation/
│   │   └── ragas_eval.py
│   │
│   └── indexing/
│       └── indexer.py
│
├── data/
│   └── documents/
│
├── storage/
│
├── .github/
│   └── workflows/
│       └── eval-and-deploy.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System health and index status |
| POST | `/ask` | Query documents using role-filtered retrieval |
| GET | `/roles` | View available roles and permissions |
| GET | `/docs` | FastAPI Swagger UI |

---

## Example Request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the digital transformation budget?",
    "role": "senior_analyst"
  }'
```

---

## Example Response

```json
{
  "answer": "Westpac is investing 1.5 billion dollars across three financial years...",
  "role": "senior_analyst",
  "allowed_classifications": [
    "PUBLIC",
    "INTERNAL",
    "CONFIDENTIAL"
  ],
  "latency_ms": 2312.67,
  "sources": [
    {
      "file": "digital_strategy.txt",
      "classification": "CONFIDENTIAL",
      "score": 0.448
    }
  ]
}
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/rosmitg/financial-doc-intelligence.git
cd financial-doc-intelligence
```

### 2. Create the environment file

```bash
cp .env.example .env
```

Fill in the required values:

```env
ANTHROPIC_API_KEY=
DATABASE_URL=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=financial-doc-intelligence
```

Do not commit your real `.env` file.

---

### 3. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

---

### 4. Install dependencies

Install CPU-only PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

### 5. Index the documents

```bash
python -m app.indexing.indexer
```

---

### 6. Run the API

```bash
python main.py
```

The API should be available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

## Docker

Build the image:

```bash
docker build -t financial-doc-intelligence .
```

Run the container:

```bash
docker run \
  -p 8000:8000 \
  --env-file .env \
  financial-doc-intelligence
```

---

## Evaluation

Run the RAGAS evaluation suite:

```bash
python -m app.evaluation.ragas_eval
```

The evaluation runs the golden dataset through the RAG pipeline and reports:

- faithfulness
- answer relevancy
- context precision
- context recall
- latency

The system also retains per-question evaluation results so individual retrieval or generation failures can be inspected rather than relying only on aggregate scores.

---

## Prompt Optimisation

DSPy is used to optimise prompt behaviour against the evaluation dataset.

Run the optimiser with:

```bash
python -m app.chains.dspy_optimizer
```

Optimised prompt artefacts are saved under:

```text
storage/
```

---

## CI Pipeline

GitHub Actions runs on pushes to `main`.

The current pipeline:

1. Installs dependencies
2. Runs the RAGAS evaluation suite
3. Checks configured quality thresholds
4. Fails the pipeline if evaluation metrics regress below thresholds
5. Builds the Docker image
6. Verifies the container health check

Cloud deployment can be added after the verified build stage.

The current repository focuses on:

> evaluation gate → build verification

rather than claiming a production deployment that does not yet exist.

---

## Reliability and Evaluation Philosophy

RAG quality is treated as an engineering problem rather than judged only by manual inspection.

The system separates:

```text
Retrieval Quality
├── Context Precision
└── Context Recall

Generation Quality
├── Faithfulness
└── Answer Relevancy
```

A fixed golden dataset allows changes to:

- prompts
- retrieval settings
- chunking
- ranking
- model configuration

to be measured against previous behaviour.

This makes RAG regressions visible before they are accepted into the main branch.

---

## Known Limitations

- The documents are sample financial data used for demonstration purposes and are not real internal financial institution documents.

- The API currently accepts a role directly from the client. A production deployment would require authentication and server-derived authorisation claims.

- The evaluation dataset currently contains only 10 labelled questions. A larger and more diverse dataset would provide stronger regression coverage.

- Retrieval may occasionally miss the best chunk because of ranking behaviour. Further hybrid-search tuning, reranking, and threshold optimisation could improve recall.

- Multi-turn conversation state is currently basic. More persistent workflow or conversation state may be added later if required.

- The system currently focuses on document intelligence rather than real-time financial data pipelines.

---

## Future Improvements

Potential extensions include:

- authenticated JWT/session-based RBAC
- larger evaluation datasets
- reranking after hybrid retrieval
- query rewriting
- retrieval failure analysis
- persistent multi-turn state
- agentic retrieval workflows
- production deployment
- monitoring dashboards
- automated regression reports
- more granular document permissions

---

## Engineering Focus

This project was built to demonstrate practical AI engineering concepts including:

- RAG architecture
- vector search
- hybrid retrieval
- metadata filtering
- permission-aware retrieval
- prompt optimisation
- LLM evaluation
- observability
- API development
- containerisation
- CI quality gates
- latency measurement
- regression testing
- grounded generation

The goal is not only to produce answers from documents, but to build a RAG system whose retrieval quality, access controls, and generation behaviour can be measured and reasoned about.
