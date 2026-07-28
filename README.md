# FinanceBench RAG

A production-oriented Retrieval-Augmented Generation (RAG) system for answering questions over SEC financial filings from the FinanceBench dataset.

The project was built to explore the software engineering principles behind modern LLM applications, with a focus on modularity, maintainability, and extensibility. Rather than being a simple prototype, it follows a layered architecture that cleanly separates retrieval, reranking, prompt construction, generation, and API serving.

---

## Features

* Dense semantic retrieval using FAISS
* Cross-encoder reranking for improved retrieval quality
* Retrieval-Augmented Generation (RAG) with OpenAI models
* Modular pipeline architecture
* FastAPI REST API
* Dependency injection and lifecycle management
* Clear separation between the core RAG pipeline and the REST API layer
* Easily extensible abstractions for retrieval, reranking, prompt building, and generation

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
       ┌─────────────┼─────────────┬─────────────┐
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
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/joao-nunes/financebench-rag.git
cd financebench-rag
```

### Option 1: Conda

```bash
conda create -n financebench-rag python 3.11.15
conda activate financebench-rag
pip install -r requirements.txt
```

### Option 2: Python virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file

```text
OPENAI_API_KEY=<your-api-key>
```

---

## Running the Application

Start the FastAPI server

```bash
uvicorn api.app:app --reload
```

The interactive API documentation is available at

```text
http://127.0.0.1:8000/docs
```

---

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
    "answer": "...",
    "sources": [
        {
            "document_id": "APPLE_2020_10K",
            "score": 0.92,
            "content": "..."
        }
    ]
}
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

The next development milestones include:

* Structured logging
* Pipeline latency metrics
* Comprehensive testing
* Docker support
* CI/CD
* Advanced retrieval techniques
* Hybrid search
* Query rewriting
* Retrieval evaluation

---

## Technologies

* Python
* FastAPI
* PyTorch
* LangChain
* FAISS
* Hugging Face Transformers
* OpenAI API
* Pydantic

---

## Motivation

This project was developed to deepen my understanding of production-oriented LLM systems and modern ML engineering practices. It complements my background in machine learning research by focusing on software architecture, maintainability, and building robust AI applications from first principles.

---

## License

This project is released under the MIT License.
