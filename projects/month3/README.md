# AgentForge — Multi-Agent Research Assistant

> **Month 3 Project** | AI Agents & Multi-Agent Systems

A production-ready multi-agent research assistant built with LangGraph. Features a supervisor-worker architecture with specialized agents (researcher, writer, critic), real web browsing, sandboxed code execution, and an MCP server for tool exposure.

## Tech Stack

- **Agent Framework:** LangGraph (StateGraph, persistence, streaming)
- **MCP:** `mcp` Python SDK (server + client)
- **Web Browsing:** httpx + BeautifulSoup + Trafilatura
- **Code Execution:** subprocess sandbox (with Docker option)
- **Observability:** LangSmith + structlog
- **LLM:** OpenAI GPT-4o / Anthropic Claude Sonnet
- **UI:** Rich terminal (live agent streaming)

## Project Structure

```
month3/
├── README.md
├── pyproject.toml
├── .env.example
├── agentforge/
│   ├── __init__.py
│   ├── state.py               # Typed state definitions
│   ├── agents/
│   │   ├── base.py            # Base agent interface
│   │   ├── researcher.py      # Web search + synthesis
│   │   ├── writer.py          # Report generation
│   │   ├── critic.py          # Fact-checking + critique
│   │   └── supervisor.py      # Task delegation + routing
│   ├── tools/
│   │   ├── registry.py        # Tool registration + discovery
│   │   ├── web_search.py      # Search tool (DuckDuckGo/Brave)
│   │   ├── web_browser.py     # Full page fetch + extract
│   │   ├── code_executor.py   # Sandboxed Python execution
│   │   ├── file_tools.py      # read_file, write_file
│   │   └── wikipedia.py       # Wikipedia API tool
│   ├── graphs/
│   │   ├── research_graph.py  # Main research workflow
│   │   ├── debate_graph.py    # Debate + verification pattern
│   │   └── subgraphs.py       # Reusable subgraph components
│   ├── mcp/
│   │   ├── server.py          # MCP server exposing tools
│   │   └── client.py          # MCP client for agent
│   └── observability/
│       ├── tracing.py         # LangSmith + OTel setup
│       └── cost_tracker.py    # Per-run token + cost tracking
├── app/
│   └── cli.py                 # Rich terminal interface
├── tests/
│   ├── test_tools.py
│   ├── test_graphs.py
│   └── evaluation/
│       └── task_suite.py      # 20 evaluation tasks
└── scripts/
    ├── run_research.py
    └── benchmark_agents.py
```

## Getting Started

### Prerequisites

```bash
python >= 3.11
# API keys: OPENAI_API_KEY or ANTHROPIC_API_KEY
# Optional: LANGCHAIN_API_KEY (LangSmith), BRAVE_API_KEY (search)
```

### Installation

```bash
cd month3/
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
```

### Usage

```bash
# Run a research task
python scripts/run_research.py "What are the key differences between RAG and fine-tuning for LLMs?"

# Start the MCP server
python -m agentforge.mcp.server --port 8765

# Run interactive CLI
python app/cli.py --mode chat

# Benchmark agent performance
python scripts/benchmark_agents.py --tasks tests/evaluation/task_suite.py
```

## Weekly Milestones

### Week 9 — LangGraph Fundamentals
**Deliverable:** A working ReAct agent with state persistence and checkpointing

- [ ] StateGraph with TypedDict state and reducers
- [ ] ReAct agent loop (think → act → observe → repeat)
- [ ] SQLite checkpointing for conversation persistence
- [ ] Human-in-the-loop approval before tool execution
- [ ] Map-reduce subgraph for parallel tool execution

### Week 10 — Tool Integration & MCP
**Deliverable:** 5+ production-quality tools + MCP server

- [ ] Tool registry with schema validation
- [ ] Web browsing tool (httpx + content extraction)
- [ ] Sandboxed Python code executor
- [ ] MCP server exposing 5+ tools
- [ ] End-to-end tool invocation through MCP protocol

### Week 11 — Multi-Agent Patterns
**Deliverable:** Supervisor-worker system with debate and verification

- [ ] Supervisor agent delegating to researcher/writer/critic workers
- [ ] Agent handoff mechanism with context passing
- [ ] Shared blackboard/memory for inter-agent communication
- [ ] Debate pattern: proposer → critic → judge
- [ ] Full research workflow: question → plan → research → synthesize → verify

### Week 12 — Reliable Agents & Production
**Deliverable:** Production-ready AgentForge with observability and evaluation

- [ ] Retry logic + fallback strategies for tool failures
- [ ] LangSmith tracing for all agent runs
- [ ] Automated evaluation: 20-task suite with scoring
- [ ] Cost tracking per agent run
- [ ] Demo video of multi-agent research pipeline

## Stretch Goals

- **Long-Horizon Tasks:** Handle 10+ step research tasks with sub-hour runtime
- **Persistent Memory:** Vector memory store that agents can query across sessions
- **Agent-as-Tool:** Use one agent as a tool for another (recursive agents)
- **Streaming UI:** Build a Gradio interface with real-time agent step visualization
- **Multi-Model:** Different agents using different LLMs (cost vs. quality optimization)

## Key Concepts

| Concept | What You'll Learn |
|---------|-------------------|
| **StateGraph** | LangGraph's graph abstraction; nodes, edges, conditional routing |
| **ReAct Pattern** | Reasoning + acting loop; why iterative is better than one-shot |
| **Checkpointing** | Agent persistence; resume interrupted tasks; debugging state |
| **MCP Protocol** | Standardized tool exposure; client-server architecture for tools |
| **Supervisor Pattern** | Task decomposition; specialist delegation; result aggregation |
| **Handoffs** | Context-preserving agent transfers; responsibility passing |
| **Debate Pattern** | Adversarial verification; reduces hallucination; improves accuracy |
