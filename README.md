# FinanceBench RAG

![CI](https://github.com/joao-nunes/financebench-rag/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Ruff](https://img.shields.io/badge/linter-ruff-D7FF64)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A production-oriented Retrieval-Augmented Generation (RAG) system for answering questions over SEC financial filings from the FinanceBench dataset.

The project was built to explore the software engineering principles behind modern LLM applications, with a focus on modularity, maintainability, and extensibility. Rather than being a simple prototype, it follows a layered architecture that cleanly separates retrieval, reranking, prompt construction, generation, and API serving.

---

## Features

* Retrieval-Augmented Generation (RAG) over SEC financial filings from the FinanceBench dataset
* Dense semantic retrieval using FAISS vector search
* Cross-encoder reranking for improved retrieval quality
* Retrieval-Augmented Generation using OpenAI models
* Modular, layered architecture with clearly separated components
* FastAPI REST API
* Dependency injection and application lifecycle management
* Docker and Docker Compose support for reproducible deployment
* Environment-based configuration using `.env`
* Easily extensible abstractions for retrieval, reranking, prompt construction, and generation
* Designed with software engineering best practices for maintainability and extensibility


---

## Architecture

The application is organized into independent components, each with a single responsibility.

```text
                        Client
                           │
                           ▼
                    FastAPI REST API
                           │
                           ▼
                      RAGService
                           │
                           ▼
                       RAGPipeline
       ┌─────────────┬─────────────┬─────────────┐
       ▼             ▼             ▼             ▼
 Retriever      Reranker    PromptBuilder   Generator
       │                                        │
       └────────────────────────────────────────┘
                           │
                           ▼
                    OpenAI Responses API
```

### Component Responsibilities

| Component         | Responsibility                                                        |
| ----------------- | --------------------------------------------------------------------- |
| **Retriever**     | Retrieves candidate documents from the FAISS vector store.            |
| **Reranker**      | Improves retrieval quality using a cross-encoder.                     |
| **PromptBuilder** | Builds the prompt from the retrieved context and the user's question. |
| **Generator**     | Calls the LLM and generates the final answer.                         |
| **RAGPipeline**   | Orchestrates the entire inference workflow.                           |
| **RAGService**    | Thin service layer used by the API.                                   |
| **FastAPI**       | Exposes the pipeline through REST endpoints.                          |

---

### Deployment

The application is containerized using Docker and orchestrated with Docker Compose. Configuration is managed through environment variables, allowing the same image to be used across local development and production environments. This ensures reproducible builds, isolated dependencies, and straightforward deployment.

## Project Structure


```text
financebench-rag/
│
├── api/
│   ├── app.py
│   ├── dependencies.py
│   ├── lifespan.py
│   ├── routes.py
│   └── schemas.py
│
├── src/
│   ├── generation/
│   ├── retrieval/
│   ├── pipeline/
│   ├── indexing/
│   ├── ingestion/
│   ├── evaluation/
│   └── config.py
│
├── scripts/
├── data/
├── tests/
├── docs/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/<your-username>/financebench-rag.git
cd financebench-rag
```

### Option 1: Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
```

### Option 2: Create a Conda environment (optional)

```bash
conda create -n financebench-rag python=3.11
conda activate financebench-rag
```

### Install the project

Project dependencies are managed through **`pyproject.toml`**.

Install the project and its runtime dependencies:

```bash
pip install .
```

For development (tests, linting, formatting, coverage, etc.):

```bash
pip install ".[dev]"
```

### Configure environment variables

Create a `.env` file (or copy `.env.example`):

```text
OPENAI_API_KEY=your_openai_api_key
EMBEDDING_DEVICE=cpu
```

### Run the API

```bash
uvicorn api.main:app --reload
```

The API will be available at:

- http://localhost:8000
- http://localhost:8000/docs

## Example Request

```http
POST /chat
```

Request

```json
{
    "question": "What was Apple's operating cash flow in 2020?"
}
```

Example Response

```json
{
  "answer": "Apple's operating cash flow in 2020 was $80.7 billion.",
  "sources": [
    {
      "document_id": "51",
      "score": 0.9910203814506531,
      "content": "Company: APPLE\nPeriod: 2018\nFiling: 10-K\nDocument ID: APPLE_2018_10K\n\nterm debt$ 102,519 $103,703 $78,927Working capital$ 14,473 $27,831 $27,863Cash generated by operating activities (2) $ 77,434 $64,225 $66,231Cash generated by/(used in) investing activities$ 16,066 $(46,446) $(45,977)Cash used in financing activities (2) $ (87,876) $(17,974) $(20,890)(1)As of September 29, 2018 , total cash, cash equivalents and marketable securities included $20.3 billion that was restricted from general use, related to the State AidDecision and other agreements.(2)Refer  to"
    },
    {
      "document_id": "12",
      "score": 0.9803022146224976,
      "content": "Company: APPLE\nPeriod: 2020\nFiling: 10-K\nDocument ID: APPLE_2020_10K\n\nterm debt$ 107,440 $102,067 $102,519 Working capital$ 38,321 $57,101 $15,410 Cash generated by operating activities$ 80,674 $69,391 $77,434 Cash generated by/(used in) investing activities$ (4,289)$45,896 $16,066 Cash used in financing activities$ (86,820)$(90,976)$(87,876)(1)As of September 26, 2020 and September 28, 2019, total marketable securities included $18.6 billion and $18.9 billion, respectively, that was restrictedfrom general use, related to the State Aid Decision (refer to Note 5,"
    },
    {
      "document_id": "2",
      "score": 0.9675090312957764,
      "content": "Company: APPLE\nPeriod: 2020\nFiling: 10-K\nDocument ID: APPLE_2020_10K\n\nminimizing the potentialrisk of principal loss. The Company’s investment policy generally requires securities to be investment grade and limits the amount of credit exposure to any oneissuer.During 2020, cash generated by operating activities of $80.7 billion was a result of $57.4 billion of net income, non-cash adjustments to net income of $17.6billion and an increase in the net change in operating assets and liabilities of $5.7 billion. Cash used in investing activities of $4.3 billion during"
    },
    {
      "document_id": "0",
      "score": 0.9208005666732788,
      "content": "Company: APPLE\nPeriod: 2020\nFiling: 10-K\nDocument ID: APPLE_2020_10K\n\nin financing activities of $86.8 billionduring 2020 consisted primarily of cash used to repurchase common stock of $72.4 billion, cash used to pay dividends and dividend equivalents of $14.1 billion,cash  used  to  repay  or  redeem  term  debt  of  $12.6  billion  and  net  repayments  of  commercial  paper  of  $1.0  billion,  partially  offset  by  net  proceeds  from  theissuance of term debt of $16.1 billion.During 2019, cash generated by operating activities of $69.4 billion was a result"
    },
    {
      "document_id": "42",
      "score": 0.9199695587158203,
      "content": "Company: APPLE\nPeriod: 2020\nFiling: 10-K\nDocument ID: APPLE_2020_10K\n\n7 %$116,914 4 %$112,093 Europe68,640 14 %60,288 (3) %62,420 Greater China40,308 (8) %43,678 (16) %51,942 Japan21,418 — %21,506 (1) %21,733 Rest of Asia Pacific19,593 10 %17,788 2 %17,407 Total net sales$ 274,515 6 %$260,174 (2) %$265,595 AmericasAmericas  net  sales  increased  during  2020  compared  to  2019  due  primarily  to  higher  net  sales  of  Services  and  Wearables,  Home  and  Accessories.  Theweakness in foreign currencies relative to the U.S. dollar had an unfavorable impact on"
    }
  ]
}
```

## Running with Docker

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=<your-api-key>
EMBEDDING_DEVICE=cpu
```

Build and start the application:

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

Interactive documentation:

```text
http://localhost:8000/docs
```

Stop the application with:

```bash
docker compose down
```


---

## Design Principles

The project follows several software engineering principles:

* Separation of concerns
* Dependency injection
* Interface-based design
* Modular architecture
* Extensibility
* Reproducibility

The retrieval pipeline is intentionally decoupled from the API layer. Internal domain models are represented using Python dataclasses, while HTTP request and response models are defined with Pydantic. This separation makes the pipeline reusable in different contexts (REST APIs, batch inference, notebooks, or CLI applications).

---

## Current Roadmap

* GitHub Actions CI/CD
* Comprehensive unit and integration testing
* Structured logging
* Pipeline latency and retrieval metrics
* Health checks and observability
* Hybrid retrieval (BM25 + dense retrieval)
* Query rewriting and expansion
* Advanced retrieval evaluation

---

## Technologies

- Python
- FastAPI
- PyTorch
- Sentence Transformers
- LangChain
- FAISS
- Hugging Face Transformers
- OpenAI API
- Docker
- Docker Compose
- Pydantic

---

## Motivation

This project was developed to deepen my understanding of production-oriented LLM systems and modern ML engineering practices. It complements my background in machine learning research by focusing on software architecture, maintainability, and building robust AI applications from first principles.

---

## License

This project is released under the MIT License.
