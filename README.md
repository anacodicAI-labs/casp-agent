# CASP Agent

Multi-agent, tool-augmented supply chain optimization system implementing the Carbon-Aware Service Performance (CASP) framework.

Supports two execution modes:

- **LLM orchestration mode**: Strands/Bedrock agents (`agents/`) calling structured `@tool` wrappers (`tools/`)
- **Python pipeline mode**: deterministic orchestration through `services/` and ML modules

Live demo: [anacodicai.com/casp](https://anacodicai.com/casp/)

---

## Repository Structure

```
casp-agent/
├── agents/          # LLM agents (Orchestrator, Risk, Sourcing)
├── tools/           # @tool wrappers called by agents
├── services/        # Deterministic business logic (orchestrator, risk, sourcing, carbon, governance)
├── ml/              # ML models (predictive analytics, vendor segmentation, early warning)
├── analytics/       # Carbon intelligence, grid scenarios, trade-off frontiers
├── config/          # Agent mapping, vehicle emissions, grid carbon, carrier/route config
├── data/
│   ├── datasets/    # Delivery_Logistics.csv (25k records)
│   ├── reference/   # vehicle_emissions.csv, grid_carbon.json, carriers.json, routes.csv
│   └── apis/        # Weather, news, distance API wrappers
├── api/             # FastAPI app (app.py, routes.py, models.py)
├── utils/           # Non-LLM helpers (extraction, risk, sourcing, carbon tools)
├── scripts/         # Data generation and evaluation scripts
├── tests/           # Test suite
├── frontend/        # UI (index.html, supply_chain.webp)
├── outputs/         # Generated outputs (figdata CSVs, model evaluation JSON, run logs)
├── app.py           # FastAPI entrypoint (deployment)
├── main.py          # CLI entrypoint
└── requirements.txt
```

---

## Setup

### 1. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Required:
- `AWS_REGION`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (or profile-based auth)
- `BEDROCK_MODEL_ID` (default: `us.anthropic.claude-3-7-sonnet-20250219-v1:0`)

Optional (fallbacks used if not set):
- `OPENWEATHERMAP_API_KEY`
- `NEWSAPI_API_KEY`
- `OPENROUTESERVICE_API_KEY`

Check your setup before running:
```bash
python3 scripts/check_env.py
```

---

## Running

### CLI (LLM orchestration)

```bash
python3 main.py "ship insulin mumbai to delhi"
```

### API server

```bash
uvicorn api.app:app --reload
```

API endpoints:
- `POST /api/extract` — extract shipment features from natural language
- `POST /api/optimize` — run full optimization pipeline
- `POST /api/chat` — LLM orchestration path

---

## Generating Outputs

All scripts are in `scripts/`. Outputs write to `outputs/`.

### Full pipeline (models + figures)

```bash
python3 scripts/run_all_paper_data.py
```

Generates:
- `outputs/model_evaluation_results.json` — model metrics (R², MAE, F1, accuracy, CV scores)
- `outputs/figdata/*.csv` — data for all figures and tables

### Model evaluation only

```bash
python3 scripts/evaluate_models.py
```

Generates `outputs/model_evaluation_results.json`.

### Figure data only

```bash
python3 scripts/generate_figure_data.py
```

Generates `outputs/figdata/`:

| File | Contents |
|------|----------|
| `feature_importance_delay.csv` | Feature importance for delay prediction |
| `feature_importance_ontime.csv` | Feature importance for on-time prediction |
| `forecast_failure_mae.csv` | MAE across prediction horizons |
| `llm_country_pivot.csv` | AI carbon by model × country |
| `llm_country_carbon.csv` | AI carbon full matrix |
| `pareto_9types.csv` | Pareto frontier across 9 package types |
| `cluster_scatter.csv` | Vendor segmentation clusters |
| `casp_by_country.csv` | CASP scores by grid scenario |
| `early_warning_computed.csv` | Early warning indicator values |
| `confusion_matrix.csv` | Delay prediction confusion matrix |
| `delay_rate_by_package_type.csv` | Delay rate per package type |

### Case study run

```bash
python3 scripts/run_casestudy_output.py
```

### Verify code–data consistency

```bash
python3 scripts/verify_consistency.py
```

---

## Testing

```bash
python3 -m pytest tests/
```

---

## Data

- **Dataset**: `data/datasets/Delivery_Logistics.csv` (25,000 records; source: [Delivery Logistics Dataset, Kaggle](https://www.kaggle.com/datasets/ayeshaseherr/delivery-logistics-dataset/data))
- **Reference data**: `data/reference/` — vehicle emission factors, grid carbon intensities, carrier configurations, route lookup table (all with provenance documentation)

---

## API Examples

```bash
# Extract features
curl -X POST http://127.0.0.1:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"query":"ship insulin from mumbai to delhi 150km"}'

# Optimize
curl -X POST http://127.0.0.1:8000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{"features":{"package_type":"pharmacy","origin":"mumbai","destination":"delhi","distance_km":150,"package_weight_kg":5}}'

# Chat (LLM path)
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"optimize delivery for insulin from mumbai to delhi"}'
```
