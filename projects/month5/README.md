# ProdRAG — Production RAG API

> **Month 5 Project** | Production AI Systems

A production-grade RAG API built for scale: FastAPI backend, Docker deployment, full observability stack (OpenTelemetry + Prometheus + Grafana), semantic caching with Redis, intelligent model routing, and load-tested to handle real traffic.

## Tech Stack

- **API:** FastAPI + Pydantic v2 + Uvicorn/Gunicorn
- **Vector DB:** ChromaDB (persistent, multi-collection)
- **Cache:** Redis (exact + semantic caching via GPTCache)
- **Observability:** OpenTelemetry, Prometheus, Grafana, Jaeger (tracing), Sentry (errors)
- **Container:** Docker + Docker Compose (multi-service)
- **Task Queue:** ARQ (async Redis queue) or Celery
- **Load Testing:** Locust
- **CI/CD:** GitHub Actions
- **Cloud:** Railway / Fly.io / AWS ECS / GCP Cloud Run

## Project Structure

```
month5/
├── README.md
├── docker-compose.yml         # Full stack: app + chroma + redis + monitoring
├── docker-compose.dev.yml     # Dev overrides
├── Dockerfile                 # Multi-stage production image
├── .github/
│   └── workflows/
│       ├── ci.yml             # Lint + test on PR
│       └── deploy.yml         # Deploy to staging/prod
├── pyproject.toml
├── .env.example
├── prodrag/
│   ├── __init__.py
│   ├── main.py                # FastAPI app factory
│   ├── config.py              # Settings (pydantic-settings)
│   ├── api/
│   │   ├── v1/
│   │   │   ├── ingest.py      # POST /v1/ingest, /v1/ingest/batch
│   │   │   ├── query.py       # POST /v1/query (streaming)
│   │   │   └── collections.py # CRUD /v1/collections
│   │   ├── auth.py            # API key + JWT middleware
│   │   └── middleware.py      # Rate limiting, CORS, request ID
│   ├── rag/
│   │   ├── pipeline.py        # Full RAG pipeline (Month 2 upgraded)
│   │   ├── router.py          # Model routing (cheap → expensive)
│   │   └── cache.py           # Semantic + exact match caching
│   ├── ingestion/
│   │   ├── worker.py          # Async ingestion job worker
│   │   └── queue.py           # ARQ task definitions
│   ├── observability/
│   │   ├── tracing.py         # OpenTelemetry setup
│   │   ├── metrics.py         # Prometheus metrics
│   │   ├── logging.py         # structlog JSON config
│   │   └── cost_tracker.py    # Per-request LLM cost
│   └── resilience/
│       ├── circuit_breaker.py # pybreaker integration
│       └── retry.py           # Tenacity retry policies
├── monitoring/
│   ├── prometheus.yml         # Scrape config
│   ├── grafana/
│   │   └── dashboards/
│   │       ├── rag_overview.json
│   │       └── cost_tracking.json
│   └── alerting/
│       └── rules.yml          # Alert conditions
├── tests/
│   ├── unit/
│   ├── integration/
│   └── load/
│       └── locustfile.py      # Load test scenarios
├── runbook.md                 # Ops runbook for common issues
└── scripts/
    ├── deploy.sh
    └── smoke_test.sh
```

## Getting Started

### Local Development (Docker Compose)

```bash
cd month5/
cp .env.example .env  # Add your API keys

# Start full stack (app + chromadb + redis + prometheus + grafana)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# API available at: http://localhost:8000
# Grafana at: http://localhost:3000 (admin/admin)
# Prometheus at: http://localhost:9090
# Jaeger at: http://localhost:16686
```

### Without Docker

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Requires Redis + ChromaDB running locally
uvicorn prodrag.main:app --reload --port 8000
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Ingest documents
curl -X POST http://localhost:8000/v1/ingest \
  -H "X-API-Key: $API_KEY" \
  -F "file=@document.pdf" \
  -F "collection=my_docs"

# Query with streaming
curl -X POST http://localhost:8000/v1/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main conclusion?", "collection": "my_docs", "stream": true}'
```

### Load Testing

```bash
cd tests/load/
locust -f locustfile.py --host=http://localhost:8000 --users=50 --spawn-rate=5
```

## Weekly Milestones

### Week 17 — API Design & Deployment
**Deliverable:** Containerized FastAPI app running in the cloud

- [ ] FastAPI with OpenAPI docs, CORS, rate limiting, request ID tracking
- [ ] Endpoints: /ingest, /query (streaming), /collections (CRUD)
- [ ] JWT + API key authentication with RBAC
- [ ] Docker multi-stage image < 500MB
- [ ] Deployed to cloud (Railway/Fly.io/ECS/Cloud Run) with HTTPS

### Week 18 — Observability & Monitoring
**Deliverable:** Full observability stack with dashboards and alerts

- [ ] structlog JSON logging with request ID correlation
- [ ] OpenTelemetry distributed tracing (custom spans for RAG steps)
- [ ] Prometheus metrics: request rate, latency (p50/p95/p99), token usage
- [ ] Grafana dashboards: RAG overview + cost tracking
- [ ] Sentry error tracking + alert rules for anomalies

### Week 19 — Cost Optimization & Caching
**Deliverable:** 40%+ cost reduction through caching and routing

- [ ] Redis exact-match cache + semantic cache (cosine similarity threshold)
- [ ] Model router: classify query complexity → select cheapest viable model
- [ ] Async batch processing for document ingestion (ARQ/Celery)
- [ ] Per-user budget limits + usage quotas enforced at API layer
- [ ] Cost analysis dashboard showing per-endpoint breakdown

### Week 20 — Scaling & Reliability
**Deliverable:** Production-hardened API passing load tests

- [ ] Load tests: 100 concurrent users, p95 latency < 5s for queries
- [ ] Circuit breaker for LLM providers (fail fast, serve cache)
- [ ] Multi-worker deployment (Gunicorn + Uvicorn, 4 workers)
- [ ] GitHub Actions CI/CD: test → build → staging → prod
- [ ] Ops runbook covering the 5 most likely incident scenarios

## Stretch Goals

- **Multi-Region:** Deploy to 2+ cloud regions with latency-based routing
- **Webhook Ingestion:** Accept document URLs via webhook for async ingestion
- **Usage Analytics:** Build a self-serve usage dashboard for API consumers
- **A/B Testing:** Route % of traffic to different RAG pipeline variants
- **SLA Dashboard:** Real-time SLA compliance tracking (uptime, latency, error rate)

## Key Concepts

| Concept | What You'll Learn |
|---------|-------------------|
| **Semantic Caching** | Cache by meaning, not exact text; trade freshness for speed |
| **Model Routing** | Not all queries need GPT-4; intelligent tiering saves 50-80% |
| **OpenTelemetry** | Vendor-neutral observability; traces, metrics, logs unified |
| **Circuit Breaker** | Fail fast when dependencies are down; prevent cascade failures |
| **Load Testing** | Find bottlenecks before users do; p99 > p50 matters |
| **Horizontal Scaling** | Stateless services scale out; state lives in external services |
| **RED Method** | Rate, Errors, Duration — the three metrics that matter for APIs |
