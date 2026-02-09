# 🧪 AI SimTest

**Open-source AI simulation testing platform — Test your chatbots with 100+ personas in minutes.**

An open-source alternative to enterprise tools like Snowglobe. Automatically generate realistic user personas, conduct multi-turn conversations with your AI chatbot, evaluate responses with multiple judges, and produce comprehensive test reports.

> **Think:** Waymo's simulation engine, but for conversational AI.

---

## ✨ Features

- **Persona Generation** — Auto-generate 10-100 diverse user personas (standard, edge-case, adversarial)
- **Multi-Turn Simulation** — Run realistic conversations with your bot via its API endpoint
- **Multi-Judge Evaluation** — Grounding, safety (PII + toxicity), quality (LLM-as-judge), relevance
- **Comprehensive Reports** — Failure patterns, risk scoring, per-persona insights, recommendations
- **Dataset Export** — JSONL, CSV, DPO pairs for fine-tuning
- **Multi-Provider LLM** — Works with OpenAI, Anthropic, Google, Ollama (local)
- **API + CLI** — Use via REST API or command line
- **CI/CD Ready** — GitHub Actions workflow included

## 🏗 Architecture

```
Input → Persona Generator → Conversation Simulator → Judge Engine → Report Generator → Export
         (LLM)              (LLM + Bot API)          (BERT + LLM)   (Analysis)        (JSONL/CSV)
```

**Core components:**

| Component | Purpose | Tech |
|-----------|---------|------|
| Persona Generator | Creates diverse test personas | GPT-4 / Claude / Ollama |
| Conversation Simulator | Runs multi-turn conversations | LiteLLM + httpx |
| Grounding Judge | Checks factual accuracy | Sentence-BERT (local, free) |
| Safety Judge | Detects PII + toxicity | Presidio + Detoxify (local, free) |
| Quality Judge | Evaluates helpfulness/clarity | LLM-as-judge |
| Relevance Judge | Checks on-topic responses | Heuristic + optional LLM |
| Report Generator | Produces actionable reports | Python analysis |
| Dataset Exporter | Exports for training/eval | JSONL, CSV, DPO |

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/your-org/ai-simtest.git
cd ai-simtest
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run via CLI

```bash
simtest run \
  --bot-endpoint "https://api.openai.com/v1/chat/completions" \
  --bot-api-key "sk-..." \
  --personas 20 \
  --max-turns 10 \
  --output ./reports
```

### 4. Run via API

```bash
# Start the server
simtest serve

# Create a simulation
curl -X POST http://localhost:8000/simulations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Bot Test",
    "bot_endpoint": "https://api.openai.com/v1/chat/completions",
    "bot_api_key": "sk-...",
    "num_personas": 20,
    "documentation": "My bot handles customer support for an e-commerce store..."
  }'

# Check status
curl http://localhost:8000/simulations/{sim_id}/status

# Get report
curl http://localhost:8000/simulations/{sim_id}/report
```

### 5. Run via Python

```python
import asyncio
from src.core.orchestrator import SimulationOrchestrator
from src.models import BotConfig, SimulationConfig

config = SimulationConfig(
    name="Customer Support Bot Test",
    bot=BotConfig(
        api_endpoint="https://api.openai.com/v1/chat/completions",
        api_key="sk-...",
    ),
    documentation="Our bot handles refunds, order tracking, and product info...",
    success_criteria=["Respond helpfully", "No PII leakage", "Stay on topic"],
    num_personas=20,
)

orchestrator = SimulationOrchestrator(config)
report = asyncio.run(orchestrator.run_simulation())

# Export
orchestrator.export_results("./reports", formats=["jsonl", "csv", "summary"])

# Access results
print(f"Pass rate: {report.summary.pass_rate:.1%}")
print(f"Recommendations: {report.recommendations}")
```

## 🐳 Docker

```bash
# Full stack with PostgreSQL + Redis
docker-compose up -d

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## 📁 Project Structure

```
ai-simtest/
├── src/
│   ├── api/           # FastAPI REST endpoints
│   ├── core/          # Orchestrator, config, logging, LLM client
│   ├── generators/    # Persona generation
│   ├── simulators/    # Conversation simulation engine
│   ├── judges/        # Grounding, safety, quality, relevance judges
│   ├── exporters/     # JSONL, CSV, DPO export
│   └── models/        # Pydantic data models
├── tests/             # Test suite
├── configs/           # Example configurations
├── .github/workflows/ # CI/CD
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## 💰 Cost Estimates

| Setup | Cost/100 Conv | Accuracy | Best For |
|-------|---------------|----------|----------|
| Free (Ollama) | $0 | 75-80% | Learning |
| Budget (GPT-3.5) | $2-3 | 85-90% | Side projects |
| **Balanced (GPT-4)** | **$6-10** | **90-95%** | **Most teams** |
| Premium | $15-20 | 95-98% | Production |

## 🧩 Extending

### Add a Custom Judge

```python
from src.judges import BaseJudge
from src.models import JudgmentResult, Severity

class BrandToneJudge(BaseJudge):
    name = "brand_tone"
    weight = 0.15

    async def evaluate(self, response, **kwargs):
        # Your custom logic here
        is_on_brand = "sorry" not in response.lower()
        return JudgmentResult(
            judge_name=self.name,
            passed=is_on_brand,
            score=1.0 if is_on_brand else 0.3,
            severity=Severity.MEDIUM,
            message="Brand tone check",
        )

# Use it
orchestrator = SimulationOrchestrator(config)
await orchestrator.setup_judges(custom_judges=[BrandToneJudge()])
```

## 🗺 Roadmap

- [x] Phase 1: Core pipeline (personas → simulation → judging → reports)
- [x] Phase 2: Multi-judge evaluation (grounding, safety, quality, relevance)
- [x] Phase 3: Export (JSONL, CSV, DPO pairs)
- [x] Phase 4: REST API + CLI
- [ ] Phase 5: Streamlit dashboard UI
- [ ] Phase 6: Iterative persona refinement
- [ ] Phase 7: Regression test suite generation
- [ ] Phase 8: W&B / LangSmith integration
- [ ] Phase 9: CI/CD simulation test action

## 📄 License

MIT

---

Built for the QA community. Contributions welcome! 🚀
