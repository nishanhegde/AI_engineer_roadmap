# CapstoneAI — Full-Stack AI Application

> **Month 6 Project** | Portfolio & Interview Preparation

An end-to-end, production-deployed AI application that integrates all skills from the 6-month roadmap: RAG-powered knowledge base, LangGraph agents, fine-tuned domain model, FastAPI backend, React frontend, and full CI/CD. This is your centerpiece portfolio project.

## Tech Stack

- **Backend:** FastAPI + LangGraph + ChromaDB + Redis
- **Frontend:** Next.js (or plain React) + Tailwind CSS + Vercel AI SDK
- **Agents:** LangGraph multi-agent orchestration
- **Fine-Tuning:** Your Month 4 domain-specific model (served via Ollama/vLLM)
- **Observability:** OpenTelemetry + Prometheus + Grafana + Sentry
- **CI/CD:** GitHub Actions (test → build → staging → production)
- **Deployment:** Docker + Cloud Run / Fly.io / Railway

## Capstone Ideas (Pick One)

| Idea | Components Used |
|------|----------------|
| **AI Research Assistant** | RAG over arXiv papers + research agent + report generator |
| **Code Review Bot** | Codebase RAG + agent + fine-tuned code model + GitHub integration |
| **Personal Knowledge Base** | Note ingestion + agent + Q&A + summary generation |
| **Domain Expert Chat** | Fine-tuned model + RAG over domain docs + safety guardrails |
| **Data Analysis Agent** | CSV/DB agent + code execution + chart generation + report writer |

## Project Structure

```
month6/
├── README.md
├── ARCHITECTURE.md            # System design document
├── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml             # PR checks: lint, test, type-check
│       ├── staging.yml        # Auto-deploy to staging on main merge
│       └── production.yml     # Manual deploy to prod with approval
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py            # FastAPI app
│   │   ├── api/
│   │   │   ├── chat.py        # WebSocket + SSE streaming
│   │   │   ├── documents.py   # Upload + manage docs
│   │   │   └── users.py       # Auth + user management
│   │   ├── agents/
│   │   │   └── workflow.py    # LangGraph agent workflow
│   │   ├── rag/
│   │   │   └── pipeline.py    # RAG pipeline (from Month 2+5)
│   │   └── models/
│   │       └── router.py      # Fine-tuned vs general model routing
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── e2e/
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── src/
│   │   ├── app/               # Next.js app router
│   │   ├── components/
│   │   │   ├── Chat.tsx       # Streaming chat interface
│   │   │   ├── DocUpload.tsx  # Document management
│   │   │   └── AgentSteps.tsx # Real-time agent visualization
│   │   └── lib/
│   │       └── api.ts         # API client
│   └── tests/
├── infrastructure/
│   ├── monitoring/
│   │   ├── grafana/
│   │   └── prometheus.yml
│   └── scripts/
│       ├── deploy.sh
│       └── smoke_test.sh
└── docs/
    ├── api.md                 # API documentation
    ├── runbook.md             # Operations runbook
    └── architecture.md        # System design walkthrough
```

## Getting Started

### Development

```bash
# Clone and configure
git clone https://github.com/YOUR_USERNAME/capstone-ai
cd capstone-ai
cp .env.example .env  # Fill in API keys

# Start all services
docker compose up

# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API docs: http://localhost:8000/docs
# Grafana: http://localhost:3001
```

### Deployment

```bash
# CI runs automatically on push; deploy to production:
gh workflow run production.yml
```

## Weekly Milestones

### Week 21 — Architecture & Core Build
**Deliverable:** Working backend with RAG + agent integration

- [ ] System architecture diagram + ADRs (Architecture Decision Records)
- [ ] FastAPI backend with auth, document management, WebSocket streaming
- [ ] LangGraph agent workflow integrated and tested
- [ ] RAG pipeline from Month 2 adapted for capstone domain
- [ ] GitHub repo with CI running from day one

### Week 22 — Polish & Deploy
**Deliverable:** Live production deployment with CI/CD

- [ ] Frontend chat UI with streaming agent steps
- [ ] Document upload, management, and search UI
- [ ] Fine-tuned model from Month 4 integrated (with routing)
- [ ] Unit (80%+ coverage) + integration + E2E tests passing
- [ ] Production deployment live at public URL

### Week 23 — System Design Interview Prep
**Deliverable:** 3 polished system design write-ups

- [ ] Design: RAG at scale (10M users, 1B documents)
- [ ] Design: Multi-tenant AI agent platform
- [ ] Design: LLM Gateway / proxy service
- [ ] Practice: 45-minute mock design interview (record yourself)
- [ ] Cheat sheet: key numbers, scaling patterns, AI-specific trade-offs

### Week 24 — Interview Prep & Launch
**Deliverable:** Job applications submitted, portfolio polished

- [ ] All 6 portfolio repos: polished READMEs, demos, badges
- [ ] LinkedIn/resume updated with AI engineering projects
- [ ] 5 STAR behavioral stories prepared and practiced
- [ ] Capstone demo video (3-5 min) recorded and shared
- [ ] 10+ AI engineering job applications submitted

## Portfolio Checklist

Before applying, ensure each of your 6 projects has:

- [ ] **Polished README** with demo GIF/screenshot, tech stack, quick start
- [ ] **Live Demo** (HuggingFace Spaces, Streamlit Cloud, or deployed URL)
- [ ] **Architecture Diagram** (even a simple ASCII diagram helps)
- [ ] **Test Suite** with CI badge showing tests pass
- [ ] **Technical Write-up** (blog post or detailed README section)

## Stretch Goals

- **Multi-User:** Full authentication with per-user document isolation
- **Billing:** Stripe integration for usage-based billing
- **Mobile:** React Native or PWA for mobile access
- **Collaboration:** Real-time multi-user document annotation
- **Open Source:** Release as an open-source project with community docs

## Interview Preparation Resources

### System Design Cheat Sheet

```
Key numbers to know:
- LLM latency: ~1s first token (GPT-4), ~0.1s (local 7B)
- Embedding: ~1ms per chunk (local), ~10ms (API)
- ChromaDB query: ~5-50ms for 1M vectors
- Redis cache hit: <1ms
- S3 upload: ~100ms for 1MB file
- p99 target for chat APIs: <3s end-to-end

Scaling patterns:
- Vector DB: shard by collection, replicate for read scale
- LLM serving: vLLM with continuous batching (10x throughput)
- Embeddings: cache aggressively (same doc → same embedding)
- Agents: async execution, timeout + circuit breaker per tool
```

### AI Concepts Flash Review

| Topic | Key Points |
|-------|-----------|
| Transformer | Attention = Q·Kᵀ/√d · V; KV cache for inference speed |
| RAG | Embed → Retrieve → Augment → Generate; RAGAS for eval |
| LoRA | ΔW = BA (r << d); train only A,B; merge at inference |
| DPO | Direct preference optimization; no reward model needed |
| Agents | ReAct = reason + act; tool calling; state persistence |
| Guardrails | Input rails (filter) + output rails (check) + NeMo |

## Key Concepts

| Concept | What You'll Learn |
|---------|-------------------|
| **Full-Stack AI** | End-to-end: from data ingestion to user-facing UI |
| **CI/CD for AI** | Testing ML systems; evaluations in CI; model versioning |
| **System Design** | Requirements → architecture → deep dive; trade-off analysis |
| **Portfolio** | What hiring managers look for; demo > explanation |
| **STAR Method** | Situation, Task, Action, Result; make projects tell a story |
| **Negotiation** | Know your worth; levels.fyi, Glassdoor for AI engineering |
