# 🤖 AI Engineer Roadmap

A hands-on, project-based **6-month learning roadmap** for experienced software developers targeting **LLM/GenAI Engineer** roles. Includes a browser-based progress tracker and 6 portfolio projects.

**Commitment:** 1–2 hours/day · 5 days/week · 120 days total

---

## 🚀 Quick Start

```bash
git clone https://github.com/nishanhegde/AI_engineer_roadmap
cd AI_engineer_roadmap
python3 -m http.server 7001
# Open http://localhost:7001
```

---

## 📅 Curriculum Overview

| Month | Focus | Project | Key Tech |
|-------|-------|---------|----------|
| **1** | LLM Fundamentals | Multi-Provider LLM CLI Tool | OpenAI/Anthropic SDKs, Pydantic, async, Ollama, typer |
| **2** | RAG | DocuQuery — Document Q&A | ChromaDB, embeddings, hybrid search, reranking, RAGAS |
| **3** | AI Agents | AgentForge — Multi-Agent Research | LangGraph, MCP, tool use, multi-agent patterns |
| **4** | Fine-Tuning | TunedAssist — Domain Assistant | LoRA/QLoRA, Unsloth, HF Transformers, guardrails |
| **5** | Production Systems | ProdRAG — Production RAG API | FastAPI, Docker, OpenTelemetry, caching, load testing |
| **6** | Portfolio & Interviews | CapstoneAI — Full-Stack AI App | System design, CI/CD, interview prep |

---

## 🗂 File Structure

```
AI_engineer_roadmap/
├── index.html            # Browser-based progress tracker app
├── daily_tasks.json      # All 120 days of tasks and resources
├── roadmap.md            # Human-readable roadmap with checkboxes
└── projects/
    ├── month1/README.md  # Multi-Provider LLM CLI Tool guide
    ├── month2/README.md  # DocuQuery — Document Q&A guide
    ├── month3/README.md  # AgentForge — Multi-Agent Research guide
    ├── month4/README.md  # TunedAssist — Fine-Tuning guide
    ├── month5/README.md  # ProdRAG — Production RAG API guide
    └── month6/README.md  # CapstoneAI — Capstone & Interview guide
```

---

## 📊 Progress Tracker

Open `index.html` via a local server (required to load `daily_tasks.json`):

```bash
python3 -m http.server 7001
# Open http://localhost:7001
```

**Features:**
- Dark-mode UI with sidebar month navigation
- Calendar grid with per-day task completion dots
- Click any day → checklist, resources, and notes
- **Step-by-step guides** — click "▶ How to do this" under any task for exact commands and instructions
- 🔥 Streak counter and per-month/overall progress bars
- "What's Next" panel — jump to your next incomplete day
- `localStorage` persistence (no account needed)
- Export / Import JSON progress backup

---

## 📚 What You'll Build

### Month 1 — Multi-Provider LLM CLI Tool
A CLI tool that talks to OpenAI, Anthropic, and Ollama with a unified interface, streaming, conversation history, cost tracking, and retry logic.
**→ [Project Guide](projects/month1/README.md)**

### Month 2 — DocuQuery: Document Q&A
Ingest PDFs, HTML, and Markdown into ChromaDB. Answer questions with inline citations using hybrid BM25 + semantic search and cross-encoder reranking. Evaluated with RAGAS.
**→ [Project Guide](projects/month2/README.md)**

### Month 3 — AgentForge: Multi-Agent Research Assistant
Supervisor-worker multi-agent system built on LangGraph. Includes a web browsing agent, sandboxed code executor, and a full MCP server. Features debate, handoff, and verification patterns.
**→ [Project Guide](projects/month3/README.md)**

### Month 4 — TunedAssist: Domain-Specific Assistant
Fine-tune a 7B LLM with LoRA/QLoRA using Unsloth. Includes LLM-as-judge evaluation, DPO preference tuning, NeMo Guardrails, and red-teaming.
**→ [Project Guide](projects/month4/README.md)**

### Month 5 — ProdRAG: Production RAG API
FastAPI + Docker RAG service with OpenTelemetry tracing, Prometheus metrics, Grafana dashboards, Redis semantic caching, intelligent model routing, and Locust load testing.
**→ [Project Guide](projects/month5/README.md)**

### Month 6 — CapstoneAI: Full-Stack AI Application
End-to-end AI app combining all prior skills: RAG + agents + fine-tuned model + React frontend + CI/CD + production deployment. Also covers 3 AI system design exercises and interview prep.
**→ [Project Guide](projects/month6/README.md)**

---

## 🎯 Skills You'll Gain

- **LLM APIs:** OpenAI, Anthropic, Ollama — unified abstractions, streaming, function calling
- **RAG:** Chunking, embeddings, vector search, hybrid retrieval, reranking, evaluation (RAGAS)
- **Agents:** LangGraph, ReAct loop, tool use, MCP protocol, multi-agent patterns
- **Fine-Tuning:** SFT, LoRA, QLoRA, DPO, evaluation, safety guardrails
- **Production:** FastAPI, Docker, observability, caching, scaling, CI/CD
- **Interviews:** System design for AI systems, coding, behavioral, portfolio

---

## 📋 Prerequisites

- Python 3.11+ (solid experience)
- Familiarity with async/await, REST APIs, and basic ML concepts
- API keys: [OpenAI](https://platform.openai.com) and/or [Anthropic](https://console.anthropic.com)
- GPU optional (Month 4 fine-tuning works with cloud Colab/Kaggle if no local GPU)

---

## 📖 Usage

### Track progress in the browser
```bash
python3 -m http.server 7001
# Open http://localhost:7001
```

### Read the full roadmap
```bash
open roadmap.md   # or any markdown viewer
```

### Start a project
```bash
cd projects/month1
cat README.md
```

---

## 🤝 Contributing

Found a broken resource link or want to suggest a better learning resource? PRs are welcome.

---

## 📄 License

MIT — use this freely for your own learning journey.
