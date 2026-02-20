# TunedAssist — Domain-Specific Assistant

> **Month 4 Project** | Fine-Tuning & Customization

A fine-tuned language model specialized for a specific domain (e.g., medical, legal, technical support, coding). Built using LoRA/QLoRA with Unsloth, evaluated with LLM-as-judge benchmarks, and protected by NeMo Guardrails.

## Tech Stack

- **Training:** HuggingFace Transformers, TRL (SFTTrainer, DPOTrainer)
- **Efficient Fine-Tuning:** PEFT (LoRA, QLoRA), bitsandbytes (4-bit)
- **Fast Training:** Unsloth (2x speedup, 70% memory reduction)
- **Evaluation:** RAGAS (for RAG tasks), custom LLM-as-judge, LM Eval Harness
- **Safety:** NeMo Guardrails, Garak (vulnerability scanner)
- **Serving:** Ollama (GGUF), vLLM
- **UI:** Gradio (with guardrails visualization)

## Project Structure

```
month4/
├── README.md
├── pyproject.toml
├── .env.example
├── data/
│   ├── raw/                   # Source data (not committed)
│   ├── processed/
│   │   ├── train.jsonl        # Training set (80%)
│   │   ├── validation.jsonl   # Validation set (10%)
│   │   └── test.jsonl         # Test set (10%)
│   └── preferences/
│       └── dpo_pairs.jsonl    # Chosen/rejected pairs for DPO
├── tunedassist/
│   ├── __init__.py
│   ├── data/
│   │   ├── formatter.py       # Chat template formatting
│   │   ├── collector.py       # Data collection utilities
│   │   └── augmentor.py       # Synthetic data generation
│   ├── training/
│   │   ├── sft_trainer.py     # Supervised fine-tuning
│   │   ├── lora_config.py     # LoRA/QLoRA configuration
│   │   ├── unsloth_trainer.py # Unsloth-accelerated training
│   │   └── dpo_trainer.py     # DPO preference training
│   ├── evaluation/
│   │   ├── llm_judge.py       # LLM-as-judge evaluation
│   │   ├── benchmark.py       # Custom domain benchmark
│   │   ├── ab_test.py         # A/B testing framework
│   │   └── metrics.py         # Scoring utilities
│   ├── serving/
│   │   ├── inference.py       # Model inference wrapper
│   │   ├── export.py          # GGUF export for Ollama
│   │   └── api.py             # FastAPI inference endpoint
│   └── safety/
│       ├── guardrails/
│       │   ├── config.yaml    # NeMo Guardrails config
│       │   └── rails.py       # Guardrails integration
│       ├── injection_detector.py
│       └── red_team.py        # Automated red-teaming
├── app/
│   └── gradio_app.py          # Gradio demo with safety UI
├── experiments/
│   └── training_logs/         # W&B experiment tracking
├── tests/
│   ├── test_formatting.py
│   ├── test_evaluation.py
│   └── test_guardrails.py
└── scripts/
    ├── prepare_data.py
    ├── train_sft.py
    ├── train_dpo.py
    ├── evaluate.py
    └── red_team.py
```

## Getting Started

### Prerequisites

```bash
python >= 3.11
# GPU recommended: 6-8GB VRAM minimum for QLoRA on 7B models
# CPU-only possible for 1-3B models
# API keys: OPENAI_API_KEY or ANTHROPIC_API_KEY (for evaluation)
# Optional: WANDB_API_KEY (experiment tracking)
```

### Installation

```bash
cd month4/
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"  # Installs transformers, peft, unsloth, etc.
cp .env.example .env
```

### Training

```bash
# 1. Prepare your dataset
python scripts/prepare_data.py --input data/raw/ --output data/processed/

# 2. SFT with LoRA (baseline)
python scripts/train_sft.py --config configs/lora_config.yaml --use-lora

# 3. SFT with QLoRA + Unsloth (faster, less memory)
python scripts/train_sft.py --config configs/qlora_config.yaml --use-unsloth

# 4. DPO training (on top of SFT checkpoint)
python scripts/train_dpo.py --base-model outputs/sft/final --data data/preferences/

# 5. Export to GGUF for Ollama
python tunedassist/serving/export.py --model outputs/dpo/final --format gguf
```

### Evaluation

```bash
# Run full benchmark suite
python scripts/evaluate.py --model outputs/dpo/final --benchmark tests/

# LLM-as-judge pairwise evaluation
python tunedassist/evaluation/llm_judge.py \
  --model-a outputs/sft/final \
  --model-b outputs/dpo/final \
  --test-set data/processed/test.jsonl

# Red-teaming
python scripts/red_team.py --model outputs/dpo/final --output results/red_team_report.json

# Launch Gradio demo
python app/gradio_app.py --model outputs/dpo/final
```

## Weekly Milestones

### Week 13 — Fine-Tuning Fundamentals
**Deliverable:** First trained model with clear improvement over base model

- [ ] Domain dataset: 500+ training examples in chat format
- [ ] SFT training run on TinyLlama or Phi-2 (proves setup works)
- [ ] Training curves analysis: identify overfitting/underfitting
- [ ] Side-by-side comparison: base vs fine-tuned on 10 test prompts
- [ ] W&B experiment tracking configured

### Week 14 — LoRA & PEFT
**Deliverable:** QLoRA-trained 7B model served via Ollama

- [ ] LoRA training with PEFT (rank experiments: 4, 8, 16, 32)
- [ ] QLoRA training on 7B model (Llama-3, Mistral, etc.)
- [ ] Unsloth training: verify ≥1.5x speedup
- [ ] Model exported to GGUF and imported into Ollama
- [ ] Adapter registry for managing multiple fine-tune checkpoints

### Week 15 — Evaluation & Benchmarking
**Deliverable:** Full evaluation report with A/B test results

- [ ] 30+ question domain benchmark with automated scoring
- [ ] LLM-as-judge evaluation (50 prompts) with rubric
- [ ] A/B test framework with blind human evaluation (20 judgments)
- [ ] DPO training on preference pairs
- [ ] Final report: base vs SFT vs SFT+DPO on all metrics

### Week 16 — Guardrails & Safety
**Deliverable:** Safe, red-teamed TunedAssist with Gradio demo

- [ ] NeMo Guardrails: input + output rails configured
- [ ] Prompt injection detection with 15 test cases
- [ ] Manual red-teaming report (20 adversarial scenarios)
- [ ] Model card with capabilities, limitations, and safety notes
- [ ] Gradio demo with real-time guardrails visualization

## Stretch Goals

- **RLHF:** Implement a simple reward model and PPO training loop
- **Model Merging:** Merge multiple LoRA adapters (e.g., coding + safety adapter)
- **Distillation:** Distill a GPT-4 or Claude fine-tune into your smaller model
- **Quantization Study:** Compare GGUF Q4_K_M vs Q8_0 vs full precision quality
- **Continuous Learning:** Set up a data flywheel to collect and incorporate user feedback

## Key Concepts

| Concept | What You'll Learn |
|---------|-------------------|
| **SFT** | Teaching desired behavior via demonstration; data quality >> quantity |
| **LoRA** | Low-rank decomposition of weight updates; efficient parameter tuning |
| **QLoRA** | 4-bit quantization + LoRA; enables 7B fine-tuning on consumer GPUs |
| **Unsloth** | Flash attention + custom kernels; 2x faster, 70% less VRAM |
| **DPO** | Train on preferences without reward model; simpler than RLHF |
| **LLM-as-Judge** | Use strong model to evaluate weak model; scalable evaluation |
| **Guardrails** | Defense-in-depth: input rails, output rails, injection detection |
