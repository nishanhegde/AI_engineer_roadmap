# Multi-Provider LLM CLI Tool

A powerful command-line interface tool that unifies access to multiple LLM providers -- OpenAI, Anthropic, and Ollama -- with streaming responses, persistent conversation history, and real-time cost tracking. Built as the foundational project for the AI Engineer Roadmap.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-SDK-412991?style=flat&logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-SDK-D4A574?style=flat)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-000000?style=flat)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat&logo=pydantic&logoColor=white)
![Typer](https://img.shields.io/badge/Typer-CLI-009688?style=flat)
![Rich](https://img.shields.io/badge/Rich-Terminal_UI-7C3AED?style=flat)
![pytest](https://img.shields.io/badge/pytest-Testing-0A9EDC?style=flat&logo=pytest&logoColor=white)

| Category | Tools |
|---|---|
| Language | Python 3.11+ |
| LLM Providers | OpenAI SDK, Anthropic SDK, Ollama |
| Validation | Pydantic v2 |
| CLI Framework | Typer |
| Terminal UI | Rich |
| Tokenization | tiktoken |
| Testing | pytest, pytest-asyncio |

---

## Project Structure

```
month1/
├── README.md
├── pyproject.toml
├── .env.example
├── src/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI entry point
│   ├── config.py               # Pydantic settings and configuration
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base provider
│   │   ├── openai_provider.py  # OpenAI API integration
│   │   ├── anthropic_provider.py  # Anthropic API integration
│   │   └── ollama_provider.py  # Ollama local model integration
│   ├── conversation/
│   │   ├── __init__.py
│   │   ├── history.py          # Conversation history management
│   │   └── storage.py          # Persistence (JSON/SQLite)
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── templates.py        # Prompt templates and engineering
│   │   └── system_prompts.py   # System prompt library
│   ├── cost/
│   │   ├── __init__.py
│   │   ├── tracker.py          # Token counting and cost calculation
│   │   └── models.py           # Pricing data per model
│   ├── output/
│   │   ├── __init__.py
│   │   ├── streaming.py        # Streaming response handler
│   │   ├── formatter.py        # Rich output formatting
│   │   └── structured.py       # Structured output parsing
│   └── utils/
│       ├── __init__.py
│       └── errors.py           # Custom exceptions and error handling
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_providers.py
│   ├── test_conversation.py
│   ├── test_cost_tracker.py
│   └── test_cli.py
└── conversations/              # Saved conversation history
    └── .gitkeep
```

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- An OpenAI API key (for GPT models)
- An Anthropic API key (for Claude models)
- Ollama installed locally (for local models)

### Installation

```bash
# Clone and navigate to the project
cd projects/month1

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Usage

```bash
# Basic chat with default provider (OpenAI)
llm-cli chat "Explain transformers in simple terms"

# Use a specific provider and model
llm-cli chat --provider anthropic --model claude-sonnet-4-20250514 "Write a haiku about code"

# Use a local model via Ollama
llm-cli chat --provider ollama --model llama3 "What is RAG?"

# Start an interactive conversation
llm-cli conversation --provider openai

# View conversation history
llm-cli history list

# Check usage costs
llm-cli cost summary

# Get structured JSON output
llm-cli chat --output json "List 3 Python frameworks with pros and cons"
```

---

## Weekly Milestones

### Week 1: Python Refresh + API Foundations

**Focus:** Reestablish Python fluency and connect to your first LLM APIs.

- [ ] Set up project structure with `pyproject.toml` and virtual environment
- [ ] Implement Pydantic configuration models for API keys and settings
- [ ] Build the abstract `BaseProvider` class with async support
- [ ] Implement `OpenAIProvider` with basic chat completions
- [ ] Implement `AnthropicProvider` with basic message creation
- [ ] Write unit tests for both providers using mocked responses
- [ ] Handle API errors gracefully (rate limits, auth failures, timeouts)

**Deliverable:** A Python script that can send prompts to both OpenAI and Anthropic and print responses.

---

### Week 2: Prompt Engineering + Conversation Memory

**Focus:** Build prompt templates and persistent conversation history.

- [ ] Create a prompt template system with variable substitution
- [ ] Build a library of reusable system prompts (coder, writer, analyst)
- [ ] Implement conversation history with message role tracking
- [ ] Add JSON-based conversation persistence (save/load sessions)
- [ ] Support multi-turn conversations with context windowing
- [ ] Implement token-aware context truncation using tiktoken
- [ ] Add conversation search and listing commands

**Deliverable:** Interactive multi-turn conversations that persist across sessions.

---

### Week 3: Structured Output + Local Models

**Focus:** Parse structured responses and integrate local LLMs via Ollama.

- [ ] Implement structured output parsing with Pydantic models
- [ ] Add JSON mode support for OpenAI and Anthropic
- [ ] Build the `OllamaProvider` for local model inference
- [ ] Add model listing and selection for all providers
- [ ] Implement response validation against expected schemas
- [ ] Add retry logic with exponential backoff for failed parses
- [ ] Compare response quality across providers for identical prompts

**Deliverable:** Structured JSON outputs from all three providers, including local models.

---

### Week 4: CLI Polish + Testing

**Focus:** Build the production-quality CLI and achieve solid test coverage.

- [ ] Build the full Typer CLI with subcommands and help text
- [ ] Add Rich-based terminal UI (syntax highlighting, tables, spinners)
- [ ] Implement streaming output with real-time token display
- [ ] Build cost tracking with per-session and cumulative reporting
- [ ] Add configuration file support (`~/.llm-cli/config.yaml`)
- [ ] Write integration tests for the full CLI workflow
- [ ] Achieve 80%+ test coverage

**Deliverable:** A polished, tested CLI tool installable via `pip install -e .`

---

## Stretch Goals

- [ ] **Plugin system** -- Allow users to add custom providers via a plugin interface
- [ ] **Response caching** -- Cache identical prompts with TTL to reduce API costs
- [ ] **Export support** -- Export conversations to Markdown, HTML, or PDF
- [ ] **Shell integration** -- Pipe stdin/stdout for use in shell scripts and pipelines
- [ ] **Model benchmarking** -- Built-in benchmarking mode that compares latency, cost, and quality across providers

---

## Key Concepts

| Concept | Description |
|---|---|
| **Provider Abstraction** | A common interface across different LLM APIs so the CLI is provider-agnostic |
| **Streaming Responses** | Server-Sent Events (SSE) for real-time token-by-token output |
| **Token Counting** | Using tiktoken to count tokens before and after requests for accurate cost tracking |
| **Prompt Engineering** | Systematic design of system prompts, few-shot examples, and template variables |
| **Structured Output** | Constraining LLM responses to match Pydantic schemas for reliable downstream use |
| **Conversation Context** | Managing message history with token-aware truncation to stay within context windows |
| **Async I/O** | Using asyncio and async providers for non-blocking API calls |
