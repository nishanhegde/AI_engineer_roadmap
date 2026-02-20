# DocuQuery — Document Q&A System

> **Month 2 Project** | RAG (Retrieval-Augmented Generation)

A production-quality document question-answering system that ingests PDFs, HTML, and Markdown files, stores embeddings in ChromaDB, and answers questions with inline citations. Covers the full RAG stack from chunking strategies to RAGAS evaluation.

## Tech Stack

- **Vector DB:** ChromaDB (persistent)
- **Embeddings:** OpenAI `text-embedding-3-small` + `sentence-transformers` (local)
- **Document Parsing:** PyMuPDF (PDF), BeautifulSoup (HTML), markdown-it-py (MD)
- **Retrieval:** Hybrid search — BM25 (`rank_bm25`) + semantic (ChromaDB)
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` + Cohere Rerank
- **Evaluation:** RAGAS (faithfulness, answer relevancy, context precision)
- **UI:** Streamlit or Gradio
- **LLM:** OpenAI GPT-4o-mini / Anthropic Claude

## Project Structure

```
month2/
├── README.md
├── pyproject.toml
├── .env.example
├── docuquery/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── chunkers.py        # Fixed, recursive, semantic chunking
│   │   ├── parsers.py         # PDF, HTML, Markdown parsers
│   │   └── pipeline.py        # End-to-end ingestion pipeline
│   ├── embeddings/
│   │   ├── providers.py       # OpenAI + local embedding providers
│   │   └── store.py           # ChromaDB wrapper
│   ├── retrieval/
│   │   ├── bm25_retriever.py  # BM25 keyword search
│   │   ├── semantic_retriever.py
│   │   ├── hybrid_retriever.py # RRF fusion
│   │   └── reranker.py        # Cross-encoder reranking
│   ├── generation/
│   │   ├── rag_pipeline.py    # Full RAG pipeline v1 + v2
│   │   ├── citations.py       # Citation extraction + verification
│   │   └── query_router.py    # Route queries to collections
│   └── evaluation/
│       ├── ragas_eval.py      # RAGAS evaluation suite
│       └── test_data.py       # Evaluation dataset
├── app/
│   └── streamlit_app.py       # Streamlit UI
├── tests/
│   ├── test_chunkers.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
└── scripts/
    ├── ingest_demo.py
    └── evaluate.py
```

## Getting Started

### Prerequisites

```bash
python >= 3.11
# API keys: OPENAI_API_KEY or ANTHROPIC_API_KEY
# Optional: COHERE_API_KEY for reranking
```

### Installation

```bash
cd month2/
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env  # Add your API keys
```

### Usage

```bash
# Ingest documents
python scripts/ingest_demo.py --path /path/to/docs --collection my_docs

# Query the system
python -m docuquery.cli query "What is the main argument in Chapter 3?" --collection my_docs

# Launch UI
streamlit run app/streamlit_app.py

# Run RAGAS evaluation
python scripts/evaluate.py --collection my_docs --output results/ragas_report.json
```

## Weekly Milestones

### Week 5 — Embeddings & Vector Fundamentals
**Deliverable:** ChromaDB collection with embedding providers and basic RAG pipeline v1

- [ ] Embedding provider abstraction (OpenAI + local sentence-transformers)
- [ ] ChromaDB persistent collection with metadata filtering
- [ ] Basic RAG pipeline: embed query → retrieve → augment → generate
- [ ] UMAP visualization of document embeddings
- [ ] Baseline retrieval precision measurement

### Week 6 — Chunking & Document Processing
**Deliverable:** Unified document processor handling PDF, HTML, and Markdown

- [ ] Three chunking strategies: fixed-size, recursive, semantic
- [ ] PDF parser with PyMuPDF (tables, headers, page metadata)
- [ ] HTML parser with BeautifulSoup + readability extraction
- [ ] Markdown parser with heading-aware chunking
- [ ] Full ingestion pipeline with deduplication and batch embedding

### Week 7 — Advanced Retrieval & Reranking
**Deliverable:** Hybrid retrieval pipeline with cross-encoder reranking

- [ ] BM25 keyword search with rank_bm25
- [ ] Hybrid search with Reciprocal Rank Fusion
- [ ] Two-stage pipeline: retrieve 20 → rerank to top 5
- [ ] HyDE query enhancement (Hypothetical Document Embeddings)
- [ ] Benchmark: v1 (basic) vs v2 (hybrid + reranking) pipeline

### Week 8 — Production RAG & Evaluation
**Deliverable:** RAGAS-evaluated DocuQuery with UI, citations, and caching

- [ ] RAGAS evaluation: faithfulness ≥ 0.8, answer relevancy ≥ 0.75
- [ ] Inline citation generation with source verification
- [ ] Multi-collection query routing
- [ ] Streamlit/Gradio UI with document upload
- [ ] Response caching for repeated queries

## Stretch Goals

- **Multimodal RAG:** Extract and embed images from PDFs (using CLIP or GPT-4V)
- **Incremental Updates:** Update document embeddings without full re-ingestion
- **Self-RAG:** Implement retrieval-on-demand (model decides when to retrieve)
- **Agentic RAG:** Use an agent that can refine queries and retrieve multiple times
- **FastAPI Wrapper:** Expose the pipeline as a REST API with authentication

## Key Concepts

| Concept | What You'll Learn |
|---------|-------------------|
| **Chunking** | Why chunk size matters; semantic vs structural splitting |
| **Embeddings** | Dense vector representations; cosine similarity; MTEB leaderboard |
| **Hybrid Search** | BM25 for keywords, semantic for concepts; RRF for fusion |
| **Reranking** | Two-stage retrieval; cross-encoders vs bi-encoders |
| **HyDE** | Generate hypothetical document, embed it, use for retrieval |
| **RAGAS** | Faithfulness, answer relevancy, context precision, recall |
| **Citations** | Grounded generation; verifiable answers reduce hallucination |
