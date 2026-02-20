# 6-Month AI Engineer Roadmap

> **Goal:** Transition to LLM/GenAI Engineer roles through hands-on, project-based learning.
> **Commitment:** 1-2 hours/day, 5 days/week = 120 days total
> **Format:** Each month builds a portfolio project, culminating in a full-stack AI capstone.

---

## Overview

| Month | Focus | Project | Key Tech |
|-------|-------|---------|----------|
| **1** | LLM Fundamentals | Multi-Provider LLM CLI Tool | OpenAI/Anthropic SDKs, Pydantic, async, Ollama, typer |
| **2** | RAG | DocuQuery - Document Q&A | ChromaDB, embeddings, chunking, hybrid search, RAGAS |
| **3** | AI Agents | AgentForge - Multi-Agent Research | LangGraph, MCP, tool use, multi-agent patterns |
| **4** | Fine-Tuning | TunedAssist - Domain Assistant | LoRA/QLoRA, Unsloth, HF Transformers, guardrails |
| **5** | Production Systems | ProdRAG - Production RAG API | FastAPI, Docker, OpenTelemetry, caching, load testing |
| **6** | Portfolio & Interview | CapstoneAI - Full-Stack AI App | System design, CI/CD, interview prep |

---

## Month 1 — LLM Fundamentals

**Project:** Multi-Provider LLM CLI Tool — Build a CLI tool supporting OpenAI, Anthropic, and Ollama with streaming, conversation history, and cost tracking.

### Week 1 — Python Refresh + API Foundations
**Concepts:** OpenAI SDK, Anthropic SDK, Pydantic, async/await, API authentication

#### Day 1 — Dev Environment & Python Async Refresh
*Time: 1.5 hours*
- [ ] Set up a Python 3.11+ virtual environment with uv or poetry
- [ ] Review Python async/await patterns: create 3 async functions that fetch data concurrently
- [ ] Install openai, anthropic, pydantic, and typer packages
- [ ] Write a Pydantic model for an LLM API request (model, messages, temperature, max_tokens)
- [ ] Verify setup by running a simple async script

**Resources:** [uv](https://docs.astral.sh/uv/) · [asyncio docs](https://docs.python.org/3/library/asyncio.html) · [Pydantic V2](https://docs.pydantic.dev/latest/)

#### Day 2 — OpenAI API Deep Dive
*Time: 1.5 hours*
- [ ] Read OpenAI API reference for chat completions endpoint
- [ ] Make your first API call using the OpenAI Python SDK
- [ ] Experiment with temperature, top_p, and max_tokens parameters
- [ ] Implement a simple multi-turn conversation (maintain message history)
- [ ] Handle API errors gracefully (rate limits, invalid requests, timeouts)

**Resources:** [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat) · [OpenAI Cookbook](https://cookbook.openai.com/)

#### Day 3 — Anthropic API Deep Dive
*Time: 1.5 hours*
- [ ] Read Anthropic API reference for messages endpoint
- [ ] Make your first Anthropic API call with claude-sonnet
- [ ] Compare Anthropic's message format with OpenAI's (system prompt handling differences)
- [ ] Implement the same multi-turn conversation using Anthropic's SDK
- [ ] Write a comparison script that sends the same prompt to both providers

**Resources:** [Anthropic API Reference](https://docs.anthropic.com/en/api/messages) · [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)

#### Day 4 — Provider Abstraction Layer
*Time: 2 hours*
- [ ] Design a unified interface (Protocol/ABC) for LLM providers
- [ ] Implement OpenAIProvider class wrapping the OpenAI SDK
- [ ] Implement AnthropicProvider class wrapping the Anthropic SDK
- [ ] Create a ProviderFactory that returns the correct provider by name
- [ ] Write unit tests for the abstraction layer using mock responses

**Resources:** [Python Protocols](https://peps.python.org/pep-0544/) · [pytest](https://docs.pytest.org/en/stable/)

#### Day 5 — Pydantic Models & Configuration
*Time: 1.5 hours*
- [ ] Create Pydantic models for: ChatMessage, ChatRequest, ChatResponse, ProviderConfig
- [ ] Implement a YAML/TOML config file loader for API keys and defaults
- [ ] Add input validation with custom Pydantic validators
- [ ] Write a configuration manager that supports env vars + config file
- [ ] Test all models with valid and invalid data

**Resources:** [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) · [12-Factor Config](https://12factor.net/config)

---

### Week 2 — Prompt Engineering & Conversation Memory
**Concepts:** system prompts, few-shot learning, chain-of-thought, conversation memory, token counting

#### Day 6 — System Prompts & Personas
*Time: 1.5 hours*
- [ ] Study effective system prompt patterns (role, constraints, output format)
- [ ] Create 5 different system prompts for different personas (coder, writer, analyst, tutor, critic)
- [ ] Implement a system prompt manager with CRUD operations
- [ ] Test how different system prompts affect output quality
- [ ] Document best practices you discover in a prompts.md file

**Resources:** [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) · [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)

#### Day 7 — Few-Shot Learning & Examples
*Time: 1.5 hours*
- [ ] Implement few-shot prompting with example injection
- [ ] Create an example store that holds reusable few-shot examples per task type
- [ ] Compare zero-shot vs few-shot performance on 3 different tasks
- [ ] Implement dynamic example selection based on input similarity
- [ ] Measure and log token usage differences between approaches

**Resources:** [Few-Shot Prompting Guide](https://www.promptingguide.ai/techniques/fewshot) · [tiktoken](https://github.com/openai/tiktoken)

#### Day 8 — Chain-of-Thought & Structured Reasoning
*Time: 1.5 hours*
- [ ] Implement chain-of-thought (CoT) prompting patterns
- [ ] Compare direct answers vs CoT on reasoning tasks (math, logic, code)
- [ ] Build a step-by-step reasoning extractor that parses CoT output
- [ ] Implement self-consistency: run same prompt N times, pick majority answer
- [ ] Create a reasoning evaluation script to score CoT quality

**Resources:** [CoT Paper](https://arxiv.org/abs/2201.11903) · [Prompting Guide - CoT](https://www.promptingguide.ai/techniques/cot)

#### Day 9 — Conversation Memory & History
*Time: 2 hours*
- [ ] Implement a ConversationMemory class with sliding window strategy
- [ ] Add token-aware truncation (keep system prompt + recent messages within limit)
- [ ] Implement conversation summarization (use LLM to summarize old messages)
- [ ] Add conversation persistence to JSON files
- [ ] Test memory strategies with a 20+ message conversation

**Resources:** [Context Window Management](https://www.anthropic.com/news/100k-context-windows) · [LangChain Memory](https://python.langchain.com/docs/concepts/memory/)

#### Day 10 — Token Counting & Cost Tracking
*Time: 1.5 hours*
- [ ] Implement token counting with tiktoken for OpenAI models
- [ ] Build a cost calculator with per-model pricing (input/output tokens)
- [ ] Create a usage tracker that logs every API call with cost
- [ ] Add daily/weekly/monthly cost reports
- [ ] Implement a budget limit feature that warns when approaching a threshold

**Resources:** [OpenAI Pricing](https://openai.com/pricing) · [Anthropic Pricing](https://www.anthropic.com/pricing)

---

### Week 3 — Structured Generation & Local Models
**Concepts:** function calling, JSON mode, streaming, Ollama, structured output

#### Day 11 — Function Calling (OpenAI)
*Time: 2 hours*
- [ ] Read OpenAI function calling documentation thoroughly
- [ ] Implement 3 tool definitions (web_search, calculator, file_reader)
- [ ] Build a tool execution loop that handles function call responses
- [ ] Handle parallel function calls and multiple tool results
- [ ] Add error handling for malformed function calls

**Resources:** [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) · [JSON Schema](https://json-schema.org/understanding-json-schema/)

#### Day 12 — Tool Use (Anthropic)
*Time: 1.5 hours*
- [ ] Read Anthropic tool use documentation
- [ ] Implement the same 3 tools using Anthropic's tool use format
- [ ] Compare OpenAI and Anthropic tool use APIs (format differences)
- [ ] Unify tool definitions to work with both providers
- [ ] Add tool use support to your provider abstraction layer

**Resources:** [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)

#### Day 13 — Structured Output & JSON Mode
*Time: 1.5 hours*
- [ ] Implement JSON mode with OpenAI (response_format)
- [ ] Use Pydantic models to validate LLM JSON output
- [ ] Build a structured data extractor (extract entities from text into Pydantic models)
- [ ] Handle partial/malformed JSON with retry logic
- [ ] Compare structured output reliability across providers

**Resources:** [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) · [Instructor Library](https://github.com/jxnl/instructor)

#### Day 14 — Streaming Responses
*Time: 1.5 hours*
- [ ] Implement streaming with OpenAI SDK (iterate over chunks)
- [ ] Implement streaming with Anthropic SDK
- [ ] Build a rich terminal display for streamed output (using rich library)
- [ ] Handle streaming errors and connection interruptions
- [ ] Add streaming support to your unified provider interface

**Resources:** [OpenAI Streaming](https://platform.openai.com/docs/api-reference/streaming) · [Rich Library](https://rich.readthedocs.io/en/latest/live.html)

#### Day 15 — Ollama & Local Models
*Time: 1.5 hours*
- [ ] Install Ollama and pull 2-3 models (llama3, mistral, phi-3)
- [ ] Make API calls to Ollama using its OpenAI-compatible endpoint
- [ ] Add OllamaProvider to your abstraction layer
- [ ] Compare local vs cloud model quality on 5 test prompts
- [ ] Benchmark local model speed and resource usage

**Resources:** [Ollama Documentation](https://ollama.com/) · [Ollama Model Library](https://ollama.com/library)

---

### Week 4 — CLI Tool Polish & Documentation
**Concepts:** typer CLI, testing, documentation, packaging, CI/CD basics

#### Day 16 — Build the CLI with Typer
*Time: 2 hours*
- [ ] Set up typer CLI with subcommands: chat, complete, config, history
- [ ] Implement the 'chat' command with interactive mode
- [ ] Add provider selection (--provider openai/anthropic/ollama) and model flags
- [ ] Implement 'config' subcommand for managing API keys and defaults
- [ ] Add rich formatting for CLI output (tables, panels, syntax highlighting)

**Resources:** [Typer Documentation](https://typer.tiangolo.com/) · [CLI Design Guidelines](https://clig.dev/)

#### Day 17 — Testing & Quality Assurance
*Time: 2 hours*
- [ ] Write unit tests for all provider classes using pytest + mocking
- [ ] Write integration tests for the CLI using typer.testing.CliRunner
- [ ] Add test fixtures for common API responses
- [ ] Set up pytest-cov for code coverage (target: 80%+)
- [ ] Fix any bugs discovered during testing

**Resources:** [pytest-mock](https://pytest-mock.readthedocs.io/en/latest/) · [Testing Typer Apps](https://typer.tiangolo.com/tutorial/testing/)

#### Day 18 — Error Handling & Robustness
*Time: 1.5 hours*
- [ ] Implement comprehensive error handling with custom exception hierarchy
- [ ] Add retry logic with exponential backoff for transient API errors
- [ ] Implement graceful degradation (fallback to local model if cloud fails)
- [ ] Add request/response logging for debugging
- [ ] Test error scenarios: network failures, invalid keys, rate limits

**Resources:** [tenacity](https://tenacity.readthedocs.io/en/latest/) · [structlog](https://www.structlog.org/en/stable/)

#### Day 19 — Documentation & Packaging
*Time: 1.5 hours*
- [ ] Write a comprehensive README.md with installation, usage, and examples
- [ ] Add inline docstrings to all public methods
- [ ] Create a pyproject.toml for the project
- [ ] Set up the project as an installable package
- [ ] Record a short demo (using asciinema or screenshots)

**Resources:** [Python Packaging Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/) · [asciinema](https://asciinema.org/)

#### Day 20 — Month 1 Review & Retrospective
*Time: 1.5 hours*
- [ ] Run all tests and ensure everything passes
- [ ] Review your code for any remaining issues or improvements
- [ ] Write a retrospective: key learnings, challenges, what you'd do differently
- [ ] Push final code to GitHub with proper .gitignore and LICENSE
- [ ] Preview Month 2 material and set up the RAG project skeleton

**Resources:** [Choose a License](https://choosealicense.com/) · [Conventional Commits](https://www.conventionalcommits.org/)

---

## Month 2 — RAG (Retrieval-Augmented Generation)

**Project:** DocuQuery - Document Q&A System — Ingest PDFs, HTML, and Markdown files, store embeddings in ChromaDB, answer questions with citations.

### Week 5 — Embeddings & Vector Fundamentals
**Concepts:** embeddings, vector databases, ChromaDB, cosine similarity, UMAP visualization

#### Day 21 — Understanding Embeddings
*Time: 1.5 hours*
- [ ] Learn what embeddings are and why they matter for RAG
- [ ] Generate embeddings using OpenAI's text-embedding-3-small model
- [ ] Compute cosine similarity between pairs of sentences
- [ ] Visualize embeddings in 2D using UMAP (10+ diverse sentences)
- [ ] Compare embedding quality across different models

**Resources:** [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) · [What Are Embeddings?](https://vickiboykis.com/what_are_embeddings/)

#### Day 22 — ChromaDB Setup & Basics
*Time: 1.5 hours*
- [ ] Install ChromaDB and create a persistent collection
- [ ] Add documents with embeddings and metadata to ChromaDB
- [ ] Query the collection with semantic search
- [ ] Experiment with distance metrics (cosine, L2, inner product)
- [ ] Implement metadata filtering on queries

**Resources:** [ChromaDB Documentation](https://docs.trychroma.com/) · [Vector DB Comparison](https://superlinked.com/vector-db-comparison)

#### Day 23 — Local Embeddings & Alternatives
*Time: 1.5 hours*
- [ ] Generate embeddings locally using sentence-transformers
- [ ] Compare local vs OpenAI embeddings on retrieval quality
- [ ] Benchmark embedding speed: local vs API (100 documents)
- [ ] Test Ollama's embedding endpoint
- [ ] Create an EmbeddingProvider abstraction (local + cloud)

**Resources:** [Sentence Transformers](https://www.sbert.net/) · [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

#### Day 24 — Similarity Search Deep Dive
*Time: 2 hours*
- [ ] Implement cosine similarity, dot product, and L2 distance from scratch
- [ ] Build a simple in-memory vector store (no external DB)
- [ ] Compare your implementation with ChromaDB results
- [ ] Implement approximate nearest neighbor with HNSW (via hnswlib)
- [ ] Benchmark exact vs approximate search on 10K+ vectors

**Resources:** [HNSW Algorithm](https://www.pinecone.io/learn/series/faiss/hnsw/) · [hnswlib](https://github.com/nmslib/hnswlib)

#### Day 25 — RAG Pipeline v1
*Time: 2 hours*
- [ ] Build a basic RAG pipeline: embed query → search ChromaDB → augment prompt → generate
- [ ] Implement source attribution (return which documents were used)
- [ ] Test with 20+ documents on a topic you know well
- [ ] Measure retrieval precision: are the right documents being found?
- [ ] Identify failure cases and document them

**Resources:** [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch) · [RAG Survey Paper](https://arxiv.org/abs/2312.10997)

---

### Week 6 — Chunking & Document Processing
**Concepts:** text chunking, PDF parsing, HTML extraction, markdown processing, metadata extraction

#### Day 26 — Chunking Strategies
*Time: 2 hours*
- [ ] Implement fixed-size chunking with overlap
- [ ] Implement recursive text splitting (by paragraph, sentence, word)
- [ ] Implement semantic chunking (split at topic boundaries using embeddings)
- [ ] Compare all three strategies on the same document
- [ ] Create a Chunker interface with configurable parameters

**Resources:** [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/) · [LangChain Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/)

#### Day 27 — PDF Document Processing
*Time: 2 hours*
- [ ] Parse PDFs with PyMuPDF (fitz) preserving structure
- [ ] Extract text, tables, and metadata from PDFs
- [ ] Handle multi-column layouts and headers/footers
- [ ] Implement page-aware chunking (chunks don't span pages)
- [ ] Test with 5+ different PDF types (academic papers, books, reports)

**Resources:** [PyMuPDF Documentation](https://pymupdf.readthedocs.io/en/latest/) · [unstructured Library](https://github.com/Unstructured-IO/unstructured)

#### Day 28 — HTML & Web Content Processing
*Time: 1.5 hours*
- [ ] Parse HTML with BeautifulSoup, extracting clean text
- [ ] Implement a web scraper that respects robots.txt
- [ ] Handle different content types: articles, docs, wikis
- [ ] Extract metadata (title, author, date, headings hierarchy)
- [ ] Implement readability-like content extraction (remove boilerplate)

**Resources:** [BeautifulSoup Docs](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) · [Trafilatura](https://trafilatura.readthedocs.io/en/latest/)

#### Day 29 — Markdown & Code Processing
*Time: 1.5 hours*
- [ ] Parse Markdown files preserving heading hierarchy
- [ ] Implement heading-aware chunking (chunks align with sections)
- [ ] Handle code blocks specially (keep code together, add language metadata)
- [ ] Process a GitHub repo's docs folder end-to-end
- [ ] Build a unified DocumentProcessor that handles PDF/HTML/MD

**Resources:** [markdown-it-py](https://github.com/executablebooks/markdown-it-py) · [Docling](https://github.com/DS4SD/docling)

#### Day 30 — Document Ingestion Pipeline
*Time: 2 hours*
- [ ] Build a full ingestion pipeline: load → extract → chunk → embed → store
- [ ] Add progress tracking for large document sets
- [ ] Implement deduplication (skip already-ingested documents by hash)
- [ ] Add batch embedding (process chunks in batches for efficiency)
- [ ] Test the pipeline with a mixed document corpus (50+ documents)

**Resources:** [Python concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) · [tqdm](https://tqdm.github.io/)

---

### Week 7 — Advanced Retrieval & Reranking
**Concepts:** hybrid search, BM25, reranking, cross-encoders, query expansion

#### Day 31 — BM25 & Keyword Search
*Time: 1.5 hours*
- [ ] Implement BM25 search using rank_bm25 library
- [ ] Index your document corpus with BM25
- [ ] Compare BM25 results with semantic search on the same queries
- [ ] Identify queries where BM25 wins vs where semantic wins
- [ ] Document the strengths and weaknesses of each approach

**Resources:** [BM25 Explained](https://www.pinecone.io/learn/okapi-bm25/) · [rank_bm25](https://github.com/dorianbrown/rank_bm25)

#### Day 32 — Hybrid Search (BM25 + Semantic)
*Time: 2 hours*
- [ ] Implement Reciprocal Rank Fusion (RRF) to merge BM25 and semantic results
- [ ] Experiment with different weight ratios (0.3/0.7, 0.5/0.5, 0.7/0.3)
- [ ] Build a HybridRetriever class that combines both approaches
- [ ] Evaluate hybrid search vs pure semantic on 20 test queries
- [ ] Implement score normalization for combining different similarity metrics

**Resources:** [Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/) · [Weaviate Hybrid Search](https://weaviate.io/blog/hybrid-search-explained)

#### Day 33 — Cross-Encoder Reranking
*Time: 1.5 hours*
- [ ] Install and use a cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)
- [ ] Implement a two-stage retrieval pipeline: retrieve 20 → rerank to top 5
- [ ] Compare results with and without reranking on 10 test queries
- [ ] Implement Cohere Rerank API as an alternative
- [ ] Benchmark reranking latency and quality trade-offs

**Resources:** [Cross-Encoders for Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html) · [Cohere Rerank](https://docs.cohere.com/docs/reranking)

#### Day 34 — Query Enhancement
*Time: 2 hours*
- [ ] Implement query expansion using LLM (generate 3 rephrasings of user query)
- [ ] Implement HyDE (Hypothetical Document Embeddings)
- [ ] Build a query router that classifies query type (factual, comparison, summary)
- [ ] Add query preprocessing: spell correction, entity recognition
- [ ] Compare retrieval quality with and without query enhancement

**Resources:** [HyDE Paper](https://arxiv.org/abs/2212.10496) · [Multi-Query Retrieval](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)

#### Day 35 — Advanced RAG Pipeline v2
*Time: 2 hours*
- [ ] Integrate hybrid search + reranking + query enhancement into one pipeline
- [ ] Implement context window packing (fit max relevant chunks in context)
- [ ] Add answer confidence scoring
- [ ] Build a comparison benchmark: v1 (basic) vs v2 (advanced) pipeline
- [ ] Document improvements with specific query examples

**Resources:** [Advanced RAG Techniques](https://arxiv.org/abs/2401.15884) · [RAG Fusion](https://github.com/Raudaschl/RAG-Fusion)

---

### Week 8 — Production RAG & Evaluation
**Concepts:** RAGAS evaluation, citations, query routing, caching, monitoring

#### Day 36 — RAG Evaluation with RAGAS
*Time: 2 hours*
- [ ] Install and set up RAGAS evaluation framework
- [ ] Create a test dataset of 20+ question-answer-context triples
- [ ] Evaluate your pipeline on: faithfulness, answer relevancy, context precision
- [ ] Compare RAGAS scores between v1 and v2 pipelines
- [ ] Identify and fix the lowest-scoring query categories

**Resources:** [RAGAS Documentation](https://docs.ragas.io/en/stable/) · [RAGAS Metrics](https://docs.ragas.io/en/stable/concepts/metrics/index.html)

#### Day 37 — Citation & Source Attribution
*Time: 1.5 hours*
- [ ] Implement inline citations in generated answers ([1], [2], etc.)
- [ ] Build a citation verification system (check if cited text exists in source)
- [ ] Add source metadata display (document name, page number, relevance score)
- [ ] Implement 'show sources' feature that displays retrieved chunks
- [ ] Test citation accuracy on 10 queries

**Resources:** [Anthropic Citation Guide](https://docs.anthropic.com/en/docs/build-with-claude/citations)

#### Day 38 — Query Routing & Multi-Index
*Time: 1.5 hours*
- [ ] Create separate ChromaDB collections for different document types
- [ ] Implement a query router that selects the best collection(s) per query
- [ ] Add a 'global search' mode that searches across all collections
- [ ] Implement collection-specific chunking strategies
- [ ] Test routing accuracy on 15 diverse queries

**Resources:** [Query Routing](https://python.langchain.com/docs/how_to/routing/) · [Semantic Router](https://github.com/aurelio-labs/semantic-router)

#### Day 39 — DocuQuery Final Assembly
*Time: 2 hours*
- [ ] Integrate all components into the DocuQuery CLI/API
- [ ] Add a Streamlit or Gradio UI for document upload and Q&A
- [ ] Implement conversation mode (follow-up questions with context)
- [ ] Add response caching for repeated queries
- [ ] Write comprehensive tests for the full pipeline

**Resources:** [Streamlit Documentation](https://docs.streamlit.io/) · [Gradio Documentation](https://www.gradio.app/docs/)

#### Day 40 — Month 2 Review & Retrospective
*Time: 1.5 hours*
- [ ] Run RAGAS evaluation on the final DocuQuery pipeline
- [ ] Write a technical blog post about your RAG learnings
- [ ] Clean up code, add docstrings, and ensure tests pass
- [ ] Push to GitHub with a demo GIF/video
- [ ] Preview Month 3 and set up the agents project skeleton

---

## Month 3 — AI Agents

**Project:** AgentForge - Multi-Agent Research Assistant — Multi-agent system using LangGraph with web search, code execution, and collaborative research capabilities.

### Week 9 — LangGraph Fundamentals
**Concepts:** StateGraph, nodes, edges, conditional routing, persistence

#### Day 41 — LangGraph Core Concepts
*Time: 2 hours*
- [ ] Install langgraph and understand the StateGraph model
- [ ] Build a simple 3-node graph: input → process → output
- [ ] Add conditional edges based on state values
- [ ] Implement state typing with TypedDict
- [ ] Visualize your graph using langgraph's built-in tools

**Resources:** [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) · [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

#### Day 42 — Agent Loop Pattern
*Time: 2 hours*
- [ ] Implement the ReAct agent loop: think → act → observe → repeat
- [ ] Build a simple agent that can use 2 tools (calculator, search stub)
- [ ] Add proper stopping conditions (max iterations, task complete)
- [ ] Implement state checkpointing for debugging
- [ ] Test the agent on 5 different multi-step tasks

**Resources:** [ReAct Pattern](https://arxiv.org/abs/2210.03629) · [LangGraph ReAct Agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)

#### Day 43 — State Management & Persistence
*Time: 1.5 hours*
- [ ] Implement complex state with nested objects and lists
- [ ] Add state reducers for accumulating results
- [ ] Set up SQLite-based checkpointing for conversation persistence
- [ ] Implement 'resume from checkpoint' functionality
- [ ] Test state persistence across process restarts

**Resources:** [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)

#### Day 44 — Streaming & Human-in-the-Loop
*Time: 2 hours*
- [ ] Implement streaming of agent steps (not just final output)
- [ ] Add human-in-the-loop approval before tool execution
- [ ] Build an interrupt-and-resume flow for sensitive actions
- [ ] Create a terminal UI that shows agent thinking in real-time
- [ ] Test with a multi-step task requiring human approval

**Resources:** [LangGraph Streaming](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/) · [Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)

#### Day 45 — Graph Patterns & Subgraphs
*Time: 2 hours*
- [ ] Implement a branching graph (parallel tool execution)
- [ ] Build a subgraph for a reusable research subtask
- [ ] Implement map-reduce pattern: split task → parallel execution → merge results
- [ ] Add graph-level error handling with retry nodes
- [ ] Benchmark single-thread vs parallel execution

**Resources:** [LangGraph Subgraphs](https://langchain-ai.github.io/langgraph/how-tos/subgraph/) · [Parallel Execution](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)

---

### Week 10 — Tool Integration & MCP
**Concepts:** custom tools, MCP protocol, web browsing, code execution, file operations

#### Day 46 — Building Custom Tools
*Time: 2 hours*
- [ ] Create 5 custom tools: web_search, read_file, write_file, run_python, wikipedia
- [ ] Implement tool input validation and error handling
- [ ] Add tool execution sandboxing for code execution
- [ ] Build a tool registry for dynamic tool discovery
- [ ] Test each tool independently with various inputs

**Resources:** [LangGraph Tool Use](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/)

#### Day 47 — Model Context Protocol (MCP) Basics
*Time: 2 hours*
- [ ] Understand MCP architecture: hosts, clients, servers
- [ ] Set up the MCP Python SDK
- [ ] Build a simple MCP server that exposes 2 tools
- [ ] Connect your agent to the MCP server as a client
- [ ] Test end-to-end tool invocation through MCP

**Resources:** [MCP Documentation](https://modelcontextprotocol.io/introduction) · [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

#### Day 48 — Web Browsing Agent
*Time: 2 hours*
- [ ] Build a web browsing tool using httpx + BeautifulSoup
- [ ] Implement content extraction and summarization
- [ ] Create a multi-step web research agent that follows links
- [ ] Add URL validation and rate limiting
- [ ] Test on 3 research tasks requiring multiple web pages

**Resources:** [httpx Documentation](https://www.python-httpx.org/)

#### Day 49 — Code Execution Agent
*Time: 2 hours*
- [ ] Build a sandboxed Python code execution tool (subprocess with timeout)
- [ ] Create a code-writing agent that generates and tests code
- [ ] Implement iterative debugging: run code → fix errors → retry
- [ ] Add output capture and analysis
- [ ] Test with 5 coding tasks of varying complexity

**Resources:** [Python subprocess](https://docs.python.org/3/library/subprocess.html)

#### Day 50 — Building an MCP Server
*Time: 2 hours*
- [ ] Build a full MCP server with 5+ tools (files, search, database, code exec, notes)
- [ ] Add resource exposure through MCP (share context/files)
- [ ] Implement prompt templates in MCP
- [ ] Test the MCP server with Claude Desktop or your own client
- [ ] Document the MCP server's capabilities and setup

**Resources:** [MCP Server Guide](https://modelcontextprotocol.io/quickstart/server) · [MCP Resources](https://modelcontextprotocol.io/docs/concepts/resources)

---

### Week 11 — Multi-Agent Patterns
**Concepts:** supervisor pattern, worker agents, handoffs, shared state, debate

#### Day 51 — Supervisor-Worker Architecture
*Time: 2 hours*
- [ ] Implement a supervisor agent that delegates to specialist workers
- [ ] Create 3 worker agents: researcher, writer, critic
- [ ] Build the routing logic for task delegation
- [ ] Implement result aggregation from multiple workers
- [ ] Test on a research task requiring all three workers

**Resources:** [Multi-Agent Supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)

#### Day 52 — Agent Handoffs & Swarm Pattern
*Time: 2 hours*
- [ ] Implement agent handoff mechanism (agent A transfers control to agent B)
- [ ] Build a swarm of 3 specialized agents with handoff rules
- [ ] Add context passing during handoffs (transfer relevant state)
- [ ] Implement handoff logging and tracing
- [ ] Test a customer support scenario with handoffs between departments

**Resources:** [Agent Handoffs](https://langchain-ai.github.io/langgraph/how-tos/agent-handoffs/) · [OpenAI Swarm](https://github.com/openai/swarm)

#### Day 53 — Shared State & Communication
*Time: 2 hours*
- [ ] Implement a shared memory/blackboard for multi-agent communication
- [ ] Build a message passing system between agents
- [ ] Create a shared knowledge base that agents can read/write
- [ ] Implement conflict resolution when agents disagree
- [ ] Test collaborative document writing with 2 agents

#### Day 54 — Debate & Verification Agents
*Time: 2 hours*
- [ ] Implement a debate pattern: proposer vs critic with a judge
- [ ] Build a fact-checking agent that verifies claims with sources
- [ ] Create a verification pipeline: generate → verify → revise
- [ ] Add confidence scoring based on verification results
- [ ] Test on 5 questions where accuracy is critical

**Resources:** [LLM Debate Paper](https://arxiv.org/abs/2305.14325) · [Constitutional AI](https://arxiv.org/abs/2212.08073)

#### Day 55 — AgentForge Research Pipeline
*Time: 2 hours*
- [ ] Combine all patterns into the AgentForge research assistant
- [ ] Implement a research workflow: question → plan → research → synthesize → verify
- [ ] Add a report generation step that produces structured output
- [ ] Implement progress tracking for long-running research tasks
- [ ] Test with 3 complex research questions

**Resources:** [GPT Researcher](https://github.com/assafelovic/gpt-researcher) · [Storm Research Agent](https://github.com/stanford-oval/storm)

---

### Week 12 — Reliable Agents & Production
**Concepts:** error recovery, observability, guardrails, evaluation, deployment

#### Day 56 — Error Recovery & Resilience
*Time: 2 hours*
- [ ] Implement retry logic for failed tool executions
- [ ] Add fallback strategies (alternative tools, simpler approaches)
- [ ] Build a self-healing agent that adapts when a tool is unavailable
- [ ] Implement timeout handling for long-running tool calls
- [ ] Test resilience by simulating tool failures

#### Day 57 — Agent Observability
*Time: 2 hours*
- [ ] Add structured logging to all agent actions
- [ ] Implement tracing with LangSmith or custom tracing
- [ ] Build a dashboard showing agent step-by-step execution
- [ ] Add cost tracking per agent run (token usage, API calls)
- [ ] Create alerts for anomalous agent behavior

**Resources:** [LangSmith Documentation](https://docs.smith.langchain.com/)

#### Day 58 — Agent Evaluation & Testing
*Time: 2 hours*
- [ ] Create a test suite of 20 tasks with expected outcomes
- [ ] Implement automated agent evaluation (task completion rate, accuracy)
- [ ] Build regression tests for critical agent paths
- [ ] Measure and optimize agent latency and cost
- [ ] Compare agent performance across different LLM backends

#### Day 59 — AgentForge Final Polish
*Time: 2 hours*
- [ ] Integrate all reliability features into AgentForge
- [ ] Add a configuration system for agent behavior tuning
- [ ] Write comprehensive documentation and usage examples
- [ ] Create demo scripts showing different agent capabilities
- [ ] Record a demo video of the multi-agent research pipeline

#### Day 60 — Month 3 Review & Retrospective
*Time: 1.5 hours*
- [ ] Run full evaluation suite on AgentForge
- [ ] Write a retrospective comparing agent patterns
- [ ] Clean up code and ensure all tests pass
- [ ] Push to GitHub with comprehensive README
- [ ] Preview Month 4 and set up fine-tuning environment

---

## Month 4 — Fine-Tuning & Customization

**Project:** TunedAssist - Domain-Specific Assistant — Fine-tune an LLM for a specific domain using LoRA/QLoRA, with evaluation benchmarks and safety guardrails.

### Week 13 — Fine-Tuning Fundamentals
**Concepts:** SFT, datasets, HF Transformers, tokenization, training loops

#### Day 61 — Fine-Tuning Theory & Setup
*Time: 1.5 hours*
- [ ] Study when and why to fine-tune vs prompt engineering vs RAG
- [ ] Set up HuggingFace Transformers and datasets libraries
- [ ] Understand the SFT (Supervised Fine-Tuning) process end-to-end
- [ ] Explore HuggingFace Hub for base models and datasets
- [ ] Install and verify GPU/MPS support (or set up cloud GPU)

**Resources:** [HuggingFace Transformers](https://huggingface.co/docs/transformers/) · [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)

#### Day 62 — Dataset Preparation
*Time: 2 hours*
- [ ] Choose a domain for your TunedAssist project
- [ ] Collect and clean 500+ training examples in chat format
- [ ] Format data into the chat template format (system/user/assistant)
- [ ] Split into train/validation/test sets (80/10/10)
- [ ] Upload your dataset to HuggingFace Hub (private)

**Resources:** [HuggingFace Datasets](https://huggingface.co/docs/datasets/) · [Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating)

#### Day 63 — Tokenization Deep Dive
*Time: 1.5 hours*
- [ ] Study BPE, WordPiece, and SentencePiece tokenization algorithms
- [ ] Analyze tokenization of your domain-specific data
- [ ] Implement a data collator for chat-format training
- [ ] Understand attention masks, padding, and truncation
- [ ] Benchmark tokenization speed for your dataset

**Resources:** [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/) · [Andrej Karpathy - Tokenization](https://www.youtube.com/watch?v=zduSFxRajkE)

#### Day 64 — First Fine-Tuning Run
*Time: 2 hours*
- [ ] Fine-tune a small model (TinyLlama or Phi-2) on your dataset using SFTTrainer
- [ ] Configure training hyperparameters (lr, batch_size, epochs, warmup)
- [ ] Monitor training loss and validation loss
- [ ] Generate sample outputs at checkpoints during training
- [ ] Compare base model vs fine-tuned model on 10 test prompts

**Resources:** [Weights & Biases Integration](https://docs.wandb.ai/guides/integrations/huggingface/)

#### Day 65 — Training Analysis & Debugging
*Time: 1.5 hours*
- [ ] Analyze training curves (loss, learning rate schedule)
- [ ] Identify and fix common issues: overfitting, underfitting, catastrophic forgetting
- [ ] Experiment with different learning rates (1e-5 to 5e-4)
- [ ] Implement early stopping based on validation loss
- [ ] Document your training experiments and results

---

### Week 14 — LoRA & PEFT
**Concepts:** LoRA, QLoRA, PEFT, Unsloth, adapter merging

#### Day 66 — LoRA Theory & Implementation
*Time: 2 hours*
- [ ] Study the LoRA paper: understand rank, alpha, and target modules
- [ ] Install PEFT library and configure LoRA for your model
- [ ] Fine-tune with LoRA: compare training speed and memory vs full fine-tuning
- [ ] Experiment with different ranks (4, 8, 16, 32) and alpha values
- [ ] Compare LoRA model quality vs full fine-tuning on your test set

**Resources:** [LoRA Paper](https://arxiv.org/abs/2106.09685) · [PEFT Documentation](https://huggingface.co/docs/peft/)

#### Day 67 — QLoRA & Memory Optimization
*Time: 2 hours*
- [ ] Study QLoRA: 4-bit quantization + LoRA
- [ ] Install bitsandbytes and configure 4-bit loading
- [ ] Fine-tune a 7B model with QLoRA (should fit in 6-8GB VRAM)
- [ ] Compare QLoRA vs LoRA quality and training speed
- [ ] Measure peak memory usage during training

**Resources:** [QLoRA Paper](https://arxiv.org/abs/2305.14314) · [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

#### Day 68 — Unsloth for Fast Training
*Time: 2 hours*
- [ ] Install Unsloth and understand its optimizations
- [ ] Fine-tune the same model using Unsloth and compare training speed
- [ ] Use Unsloth's built-in data formatting helpers
- [ ] Export the model in different formats (GGUF, HF format)
- [ ] Compare Unsloth vs standard HF training on speed and quality

**Resources:** [Unsloth Documentation](https://github.com/unslothai/unsloth) · [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

#### Day 69 — Adapter Management & Merging
*Time: 1.5 hours*
- [ ] Train multiple LoRA adapters for different sub-tasks
- [ ] Implement adapter switching at inference time
- [ ] Merge LoRA weights into the base model
- [ ] Compare merged model vs adapter-based inference (speed, quality)
- [ ] Build an adapter registry for managing multiple fine-tunes

**Resources:** [PEFT Adapter Merging](https://huggingface.co/docs/peft/developer_guides/model_merging)

#### Day 70 — Serving Fine-Tuned Models
*Time: 2 hours*
- [ ] Serve your fine-tuned model with Ollama (convert to GGUF → import)
- [ ] Set up vLLM for production-grade serving
- [ ] Compare serving options: Ollama vs vLLM vs HF TGI
- [ ] Benchmark inference speed and throughput
- [ ] Create an API endpoint for your fine-tuned model

**Resources:** [vLLM Documentation](https://docs.vllm.ai/en/latest/) · [Ollama Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

---

### Week 15 — Evaluation & Benchmarking
**Concepts:** LLM-as-judge, benchmarks, A/B testing, human evaluation, metrics

#### Day 71 — LLM-as-Judge Evaluation
*Time: 2 hours*
- [ ] Implement LLM-as-judge evaluation using a strong model as evaluator
- [ ] Create evaluation rubrics for your domain (accuracy, helpfulness, safety)
- [ ] Build a pairwise comparison framework (model A vs model B)
- [ ] Run evaluation on 50 test prompts with automated scoring
- [ ] Analyze inter-rater reliability between LLM judge and your manual scoring

**Resources:** [LLM-as-Judge Paper](https://arxiv.org/abs/2306.05685) · [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)

#### Day 72 — Custom Benchmarks
*Time: 2 hours*
- [ ] Create a domain-specific benchmark suite (30+ test cases)
- [ ] Implement automated scoring for your benchmark
- [ ] Run the benchmark against base model, fine-tuned, and GPT-4/Claude
- [ ] Generate a comparison report with tables and charts
- [ ] Identify remaining weaknesses and create targeted training data

**Resources:** [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

#### Day 73 — A/B Testing Framework
*Time: 2 hours*
- [ ] Build an A/B testing framework for comparing model versions
- [ ] Implement blind evaluation (hide which model generated which response)
- [ ] Create a simple web UI for human evaluation of A/B tests
- [ ] Collect and analyze 20+ human judgments
- [ ] Calculate statistical significance of A/B test results

#### Day 74 — Iterative Improvement
*Time: 2 hours*
- [ ] Analyze evaluation results to identify failure modes
- [ ] Create targeted training data for weak areas (data augmentation)
- [ ] Re-train with the improved dataset
- [ ] Run evaluation again and compare with previous version
- [ ] Document the improvement cycle and results

**Resources:** [Data-Centric AI](https://dcai.csail.mit.edu/) · [Synthetic Data Generation](https://huggingface.co/blog/synthetic-data-save-costs)

#### Day 75 — DPO & Preference Tuning
*Time: 2 hours*
- [ ] Study DPO (Direct Preference Optimization) theory
- [ ] Create a preference dataset (chosen vs rejected pairs)
- [ ] Run DPO training on your fine-tuned model
- [ ] Compare SFT-only vs SFT+DPO on evaluation benchmarks
- [ ] Document when DPO helps vs when it doesn't

**Resources:** [DPO Paper](https://arxiv.org/abs/2305.18290) · [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)

---

### Week 16 — Guardrails & Safety
**Concepts:** guardrails, prompt injection, content filtering, red-teaming, safety

#### Day 76 — NeMo Guardrails Setup
*Time: 2 hours*
- [ ] Install NVIDIA NeMo Guardrails
- [ ] Create guardrails config for your TunedAssist domain
- [ ] Implement input rails (block harmful queries, detect off-topic)
- [ ] Implement output rails (ensure responses are safe and on-topic)
- [ ] Test guardrails with 20 adversarial prompts

**Resources:** [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)

#### Day 77 — Prompt Injection Defense
*Time: 2 hours*
- [ ] Study common prompt injection techniques
- [ ] Implement input sanitization and injection detection
- [ ] Build a prompt injection classifier using a fine-tuned model
- [ ] Test your defenses against 15 known injection patterns
- [ ] Create a defense-in-depth strategy

**Resources:** [Prompt Injection Attacks](https://simonwillison.net/2023/Apr/14/worst-that-can-happen/) · [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

#### Day 78 — Red-Teaming Your Model
*Time: 2 hours*
- [ ] Develop a red-teaming checklist for your domain
- [ ] Perform manual red-teaming (20 adversarial scenarios)
- [ ] Use an LLM to generate additional red-team prompts
- [ ] Document all vulnerabilities found
- [ ] Implement fixes for each vulnerability

**Resources:** [Red-Teaming LLMs](https://huggingface.co/blog/red-teaming) · [Garak - LLM Scanner](https://github.com/leondz/garak)

#### Day 79 — TunedAssist Final Assembly
*Time: 2 hours*
- [ ] Integrate guardrails into the TunedAssist inference pipeline
- [ ] Add a feedback collection mechanism for continuous improvement
- [ ] Write comprehensive documentation (model card, usage guide)
- [ ] Create a Gradio demo with guardrails visualization
- [ ] Run final evaluation and red-teaming pass

**Resources:** [Model Cards](https://huggingface.co/docs/hub/model-cards)

#### Day 80 — Month 4 Review & Retrospective
*Time: 1.5 hours*
- [ ] Run final benchmarks: base vs fine-tuned vs fine-tuned+DPO
- [ ] Write a model card documenting capabilities, limitations, and safety
- [ ] Clean up code and push to GitHub
- [ ] Write a retrospective on fine-tuning learnings
- [ ] Preview Month 5 and set up the production project

---

## Month 5 — Production AI Systems

**Project:** ProdRAG - Production RAG API — Production-grade RAG API with monitoring, caching, load testing, and CI/CD.

### Week 17 — API Design & Deployment
**Concepts:** FastAPI, Docker, cloud deployment, API design, authentication

#### Day 81 — FastAPI Application Structure
*Time: 2 hours*
- [ ] Set up FastAPI project with proper directory structure
- [ ] Implement health check, version, and docs endpoints
- [ ] Create Pydantic models for API request/response schemas
- [ ] Add CORS, rate limiting, and request validation middleware
- [ ] Write OpenAPI documentation with examples

**Resources:** [FastAPI Documentation](https://fastapi.tiangolo.com/) · [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)

#### Day 82 — RAG API Endpoints
*Time: 2 hours*
- [ ] Implement /ingest endpoint (upload and process documents)
- [ ] Implement /query endpoint (RAG query with citations)
- [ ] Implement /collections endpoint (manage document collections)
- [ ] Add streaming response support for long answers
- [ ] Write integration tests for all endpoints

#### Day 83 — Authentication & Security
*Time: 2 hours*
- [ ] Implement API key authentication
- [ ] Add JWT-based user authentication
- [ ] Set up role-based access control (admin, user, readonly)
- [ ] Implement input sanitization for security
- [ ] Write security tests

**Resources:** [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/) · [OWASP API Security](https://owasp.org/www-project-api-security/)

#### Day 84 — Docker & Containerization
*Time: 2 hours*
- [ ] Write a multi-stage Dockerfile for the FastAPI app
- [ ] Create docker-compose.yml with app + ChromaDB + Redis services
- [ ] Implement health checks in Docker Compose
- [ ] Optimize Docker image size (target under 500MB)
- [ ] Test the full stack locally with docker-compose up

**Resources:** [Docker Best Practices](https://docs.docker.com/build/building/best-practices/) · [Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)

#### Day 85 — Cloud Deployment
*Time: 2 hours*
- [ ] Choose a deployment target (Railway, Fly.io, AWS ECS, or GCP Cloud Run)
- [ ] Deploy the containerized application
- [ ] Set up environment variables and secrets management
- [ ] Configure HTTPS
- [ ] Test the deployed API with curl and your test suite

**Resources:** [Railway Deployment](https://docs.railway.app/) · [Fly.io Documentation](https://fly.io/docs/)

---

### Week 18 — Observability & Monitoring
**Concepts:** OpenTelemetry, metrics, logging, tracing, alerting

#### Day 86 — Structured Logging
*Time: 1.5 hours*
- [ ] Set up structlog with JSON output format
- [ ] Add request ID tracking across the full request lifecycle
- [ ] Implement log levels appropriately
- [ ] Add LLM-specific logging (model, tokens, latency, cost)
- [ ] Set up log aggregation

**Resources:** [structlog Documentation](https://www.structlog.org/en/stable/)

#### Day 87 — OpenTelemetry Tracing
*Time: 2 hours*
- [ ] Install OpenTelemetry Python SDK
- [ ] Instrument FastAPI with automatic tracing
- [ ] Add custom spans for RAG pipeline steps
- [ ] Set up Jaeger or Zipkin for trace visualization
- [ ] Trace a full RAG query and analyze the waterfall view

**Resources:** [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)

#### Day 88 — Metrics & Dashboards
*Time: 2 hours*
- [ ] Expose Prometheus metrics from FastAPI
- [ ] Add custom metrics: token usage, retrieval quality, cache hit rate
- [ ] Set up Grafana dashboard with key metrics panels
- [ ] Create a dashboard showing RAG pipeline performance breakdown
- [ ] Add SLA tracking (p50, p95, p99 latency targets)

**Resources:** [Prometheus Python Client](https://github.com/prometheus/client_python) · [Grafana Documentation](https://grafana.com/docs/grafana/latest/)

#### Day 89 — Alerting & Error Tracking
*Time: 1.5 hours*
- [ ] Set up error tracking with Sentry
- [ ] Configure alerts for: high error rate, slow responses, budget exceeded
- [ ] Implement a health check dashboard showing component status
- [ ] Create a runbook for common alert scenarios

**Resources:** [Sentry Python SDK](https://docs.sentry.io/platforms/python/)

#### Day 90 — LLM-Specific Observability
*Time: 2 hours*
- [ ] Track token usage per request with cost attribution
- [ ] Implement quality monitoring (detect low-quality or hallucinated responses)
- [ ] Build a feedback loop for flagging bad responses
- [ ] Create a daily report of usage, costs, and quality metrics
- [ ] Set up anomaly detection for unusual usage patterns

**Resources:** [LangFuse - LLM Observability](https://langfuse.com/docs)

---

### Week 19 — Cost Optimization & Caching
**Concepts:** semantic caching, model routing, batching, cost analysis, optimization

#### Day 91 — Semantic Caching
*Time: 2 hours*
- [ ] Implement exact-match caching with Redis
- [ ] Build a semantic cache: cache responses for semantically similar queries
- [ ] Configure cache TTL and invalidation strategies
- [ ] Measure cache hit rates and latency improvement
- [ ] Handle cache coherency when documents are updated

**Resources:** [GPTCache](https://github.com/zilliztech/GPTCache)

#### Day 92 — Model Routing & Tiering
*Time: 2 hours*
- [ ] Implement a model router that selects the cheapest suitable model per query
- [ ] Create tiers: simple → medium → complex model selection
- [ ] Build a complexity classifier to route queries
- [ ] Implement fallback chains (try cheap model first, escalate if needed)
- [ ] Measure cost savings from intelligent routing

**Resources:** [FrugalGPT Paper](https://arxiv.org/abs/2305.05176)

#### Day 93 — Batch Processing & Async
*Time: 2 hours*
- [ ] Implement request batching for embedding generation
- [ ] Build an async pipeline for document ingestion
- [ ] Add background job processing with Celery or ARQ
- [ ] Implement priority queues for different request types
- [ ] Benchmark throughput with batching vs sequential processing

**Resources:** [ARQ - Async Task Queue](https://arq-docs.helpmanual.io/)

#### Day 94 — Cost Analysis & Budgeting
*Time: 1.5 hours*
- [ ] Build a cost analysis dashboard showing per-endpoint costs
- [ ] Implement per-user budget limits and usage quotas
- [ ] Create cost projections based on current usage trends
- [ ] Identify the most expensive operations and optimize them
- [ ] Write a cost optimization report with recommendations

#### Day 95 — Response Quality Optimization
*Time: 2 hours*
- [ ] Implement adaptive retrieval (skip RAG for simple questions)
- [ ] Add response post-processing (format checking, fact verification)
- [ ] Implement streaming with early termination for simple answers
- [ ] Build an A/B testing framework for prompt variations
- [ ] Optimize prompt templates for lower token usage

---

### Week 20 — Scaling & Reliability
**Concepts:** load testing, circuit breakers, task queues, horizontal scaling, disaster recovery

#### Day 96 — Load Testing
*Time: 2 hours*
- [ ] Set up Locust for load testing the API
- [ ] Write load test scenarios: normal traffic, peak traffic, spike traffic
- [ ] Run load tests and identify bottlenecks
- [ ] Test concurrent document ingestion under load
- [ ] Generate a load test report with recommendations

**Resources:** [Locust Documentation](https://docs.locust.io/en/stable/)

#### Day 97 — Circuit Breakers & Resilience
*Time: 2 hours*
- [ ] Implement circuit breaker pattern for LLM API calls
- [ ] Add bulkhead pattern to isolate failures between endpoints
- [ ] Implement graceful degradation (serve cached results when LLM is down)
- [ ] Add request queuing for backpressure management
- [ ] Test resilience by simulating provider outages

**Resources:** [Circuit Breaker Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)

#### Day 98 — Horizontal Scaling
*Time: 2 hours*
- [ ] Configure multiple API workers with Gunicorn/Uvicorn
- [ ] Externalize all state (sessions, cache, vectors) to shared services
- [ ] Test scaling from 1 to 4 workers and measure throughput increase
- [ ] Document the scaling architecture

#### Day 99 — ProdRAG Final Assembly
*Time: 2 hours*
- [ ] Integrate all production features: caching, monitoring, scaling, resilience
- [ ] Write a comprehensive ops runbook
- [ ] Create a CI/CD pipeline with GitHub Actions
- [ ] Set up staging and production environments
- [ ] Run final load tests and verify all monitoring works

**Resources:** [GitHub Actions](https://docs.github.com/en/actions)

#### Day 100 — Month 5 Review & Retrospective
*Time: 1.5 hours*
- [ ] Run full load test suite and publish results
- [ ] Verify all monitoring dashboards and alerts are working
- [ ] Write a production readiness review document
- [ ] Push final code to GitHub with deployment documentation
- [ ] Preview Month 6 and plan the capstone project

---

## Month 6 — Portfolio & Interview Prep

**Project:** CapstoneAI - Full-Stack AI Application — End-to-end AI app combining RAG + agents + fine-tuning with CI/CD and production deployment.

### Week 21 — Capstone Architecture & Build
**Concepts:** system design, architecture, integration, end-to-end development

#### Day 101 — Capstone Planning & Architecture
*Time: 2 hours*
- [ ] Choose your capstone project idea (combines RAG + agents + fine-tuning)
- [ ] Draw a system architecture diagram (components, data flow, APIs)
- [ ] Define API contracts and data models
- [ ] Create a project plan with milestones for the next 2 weeks
- [ ] Set up the project repository with CI/CD from day one

**Resources:** [System Design Primer](https://github.com/donnemartin/system-design-primer) · [C4 Model](https://c4model.com/)

#### Day 102 — Backend Core Implementation
*Time: 2 hours*
- [ ] Implement the core backend with FastAPI
- [ ] Set up database models and migrations
- [ ] Implement the RAG pipeline for the capstone
- [ ] Add authentication and user management
- [ ] Write initial API tests

#### Day 103 — Agent Integration
*Time: 2 hours*
- [ ] Integrate LangGraph agents into the capstone backend
- [ ] Build agent workflows specific to your capstone domain
- [ ] Add tool integrations (search, code execution, data analysis)
- [ ] Implement WebSocket for real-time agent streaming
- [ ] Test agent workflows end-to-end

#### Day 104 — Frontend Development
*Time: 2 hours*
- [ ] Build a frontend UI (React, Next.js, or simple HTML/JS)
- [ ] Implement the chat interface with streaming display
- [ ] Add document upload and management UI
- [ ] Implement user authentication flow
- [ ] Test frontend-backend integration

**Resources:** [Vercel AI SDK](https://sdk.vercel.ai/docs) · [Tailwind CSS](https://tailwindcss.com/docs)

#### Day 105 — Fine-Tuned Model Integration
*Time: 2 hours*
- [ ] Integrate your fine-tuned model from Month 4 (or deploy via API)
- [ ] Implement model routing: use fine-tuned model for domain tasks, general model for others
- [ ] Add guardrails from Month 4 to the production pipeline
- [ ] Test the integrated system with domain-specific queries
- [ ] Benchmark: fine-tuned vs general model on capstone tasks

---

### Week 22 — Capstone Polish & Deploy
**Concepts:** testing, CI/CD, deployment, documentation, demo

#### Day 106 — Comprehensive Testing
*Time: 2 hours*
- [ ] Write unit tests for all core modules (target 80% coverage)
- [ ] Write integration tests for API endpoints
- [ ] Add end-to-end tests for critical user flows
- [ ] Implement RAG evaluation on your capstone's data
- [ ] Fix all bugs discovered during testing

#### Day 107 — CI/CD Pipeline
*Time: 2 hours*
- [ ] Set up GitHub Actions for CI (lint, test, type-check on every PR)
- [ ] Add automated deployment to staging on merge to main
- [ ] Implement deployment to production with manual approval
- [ ] Add Docker image building and pushing to registry
- [ ] Test the full CI/CD pipeline end-to-end

**Resources:** [GitHub Actions CI/CD](https://docs.github.com/en/actions/automating-builds-and-tests)

#### Day 108 — Production Deployment
*Time: 2 hours*
- [ ] Deploy the full capstone to production
- [ ] Set up monitoring and alerting (reuse Month 5 patterns)
- [ ] Configure auto-scaling (if using cloud provider)
- [ ] Set up backup and disaster recovery
- [ ] Verify production deployment with smoke tests

#### Day 109 — Documentation & Demo
*Time: 2 hours*
- [ ] Write a polished README with architecture diagram, screenshots, and GIFs
- [ ] Create API documentation (auto-generated from FastAPI + custom guides)
- [ ] Record a 3-5 minute demo video walkthrough
- [ ] Write a technical blog post about the capstone architecture
- [ ] Add badges (CI status, coverage, license) to README

**Resources:** [OBS Studio for Recording](https://obsproject.com/) · [Shields.io Badges](https://shields.io/)

#### Day 110 — Capstone Review & Refinement
*Time: 2 hours*
- [ ] Get feedback from peers or the community on your capstone
- [ ] Fix any UX issues or bugs reported
- [ ] Optimize performance based on monitoring data
- [ ] Finalize all documentation
- [ ] Submit the capstone to HuggingFace Spaces or a live URL

---

### Week 23 — System Design for AI Interviews
**Concepts:** system design, scalability, trade-offs, AI architecture, interview patterns

#### Day 111 — AI System Design Fundamentals
*Time: 2 hours*
- [ ] Study the AI system design interview framework
- [ ] Learn common AI system components (serving, feature stores, model registries)
- [ ] Practice: Design a real-time content moderation system
- [ ] Draw architecture diagrams for your design
- [ ] Identify and discuss trade-offs in your design

**Resources:** [ML System Design (Chip Huyen)](https://huyenchip.com/machine-learning-systems-design/toc.html) · [Designing ML Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

#### Day 112 — Design Exercise: RAG at Scale
*Time: 2 hours*
- [ ] Design a RAG system that serves 10M users with 1B documents
- [ ] Address: indexing pipeline, serving architecture, caching strategy
- [ ] Handle: multi-tenancy, access control, freshness guarantees
- [ ] Calculate infrastructure costs and propose optimizations
- [ ] Practice explaining your design within a time limit

#### Day 113 — Design Exercise: AI Agent Platform
*Time: 2 hours*
- [ ] Design a multi-tenant AI agent platform (like managed LangGraph)
- [ ] Address: agent isolation, resource limits, billing, observability
- [ ] Handle: tool sandboxing, state persistence, horizontal scaling
- [ ] Design the API and developer experience
- [ ] Practice the 45-minute design interview format

#### Day 114 — Design Exercise: LLM Gateway
*Time: 2 hours*
- [ ] Design an LLM Gateway/Proxy service (routing, caching, rate limiting, observability)
- [ ] Address: provider failover, cost optimization, model routing
- [ ] Handle: streaming, function calling proxy, token counting
- [ ] Design a usage analytics and billing system
- [ ] Write up your design as a 2-page architecture document

**Resources:** [LiteLLM - LLM Gateway](https://github.com/BerriAI/litellm)

#### Day 115 — Design Review & Mock Interviews
*Time: 2 hours*
- [ ] Review all 3 design exercises and refine weak areas
- [ ] Practice explaining designs within time limits
- [ ] Prepare for common follow-up questions (scaling, failure modes, cost)
- [ ] Create a cheat sheet of key numbers (latency, throughput, costs)
- [ ] Do a mock interview with a peer or rubber duck

---

### Week 24 — Technical Interview Prep & Portfolio
**Concepts:** coding interviews, behavioral interviews, portfolio, networking, job search

#### Day 116 — AI/ML Coding Interview Prep
*Time: 2 hours*
- [ ] Practice 5 LLM-related coding problems (implement RAG from scratch, build a simple agent)
- [ ] Practice 3 data structure problems relevant to AI
- [ ] Write clean, well-tested solutions with proper error handling
- [ ] Time yourself: aim for 30-45 minutes per problem
- [ ] Review and optimize your solutions

**Resources:** [NeetCode](https://neetcode.io/)

#### Day 117 — AI Concepts Deep Review
*Time: 2 hours*
- [ ] Review transformer architecture (attention, positional encoding, KV cache)
- [ ] Study: RLHF, DPO, Constitutional AI, model distillation
- [ ] Review: embeddings, vector search, RAG patterns, evaluation metrics
- [ ] Prepare concise explanations for each concept
- [ ] Create flashcards for key concepts and formulas

**Resources:** [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · [LLM Course](https://github.com/mlabonne/llm-course)

#### Day 118 — Behavioral Interview Prep
*Time: 1.5 hours*
- [ ] Prepare 5 STAR stories from your AI project experience
- [ ] Practice explaining each portfolio project in 2 minutes
- [ ] Prepare answers for: 'Why AI engineering?', 'Hardest technical challenge?'
- [ ] Practice discussing trade-offs you made in your projects
- [ ] Record yourself answering 3 behavioral questions and review

#### Day 119 — Portfolio Polish & Online Presence
*Time: 1.5 hours*
- [ ] Ensure all 6 GitHub repos have polished READMEs with demos
- [ ] Update your LinkedIn/resume with AI engineering projects and skills
- [ ] Write a portfolio summary page or personal site update
- [ ] Prepare a 'portfolio walkthrough' presentation (5 minutes)
- [ ] Connect with AI engineering communities

#### Day 120 — Final Review & Next Steps
*Time: 1.5 hours*
- [ ] Review everything you've built over 6 months — celebrate your progress!
- [ ] Identify your strongest areas and areas for continued growth
- [ ] Create a 'continued learning' plan for the next 3 months
- [ ] Start applying to AI engineering roles
- [ ] Set up informational interviews with AI engineers at target companies

**Resources:** [AI Engineer Job Boards](https://aijobs.net/) · [Levels.fyi AI Roles](https://www.levels.fyi/t/ai-engineer)

---

*Generated from `daily_tasks.json` · Track your progress at `index.html`*
