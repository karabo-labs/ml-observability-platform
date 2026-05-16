<p align="center">
  <h1 align="center">🧠 ML Observability Platform</h1>
  <p align="center">
    End-to-end MLOps pipeline with real-time model monitoring and drift detection.
    <br />
    <strong>Train → Deploy → Monitor → Retrain</strong>
    <br />
    <em>$0 infrastructure. HuggingFace + GitHub Actions.</em>
  </p>
</p>

## 📋 Overview

A complete MLOps lifecycle demo that shows you're not just a model trainer — you build **production ML systems**.

| Layer | Technology | Cost |
|-------|-----------|------|
| Model Training | GitHub Actions (free runner) | $0 |
| Model Registry | Hugging Face Hub | $0 |
| Inference API | Hugging Face Spaces (Gradio) | $0 |
| Monitoring Dashboard | Same Space — 2nd tab | $0 |
| Drift Detection | PSI + statistical tests | $0 |
| CI/CD | GitHub Actions → HF Deploy | $0 |

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     GitHub Actions (CI/CD)                   │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │  Train   │───▶│   Push   │───▶│  Deploy  │               │
│  │  Model   │    │  to Hub  │    │  Space   │               │
│  └──────────┘    └──────────┘    └────┬─────┘               │
│       │                               │                     │
│       ▼                               ▼                     │
│  ┌──────────┐                  ┌──────────────┐             │
│  │  Tests   │                  │  Health      │             │
│  │  + Drift │                  │  Check       │             │
│  └──────────┘                  └──────────────┘             │
└──────────────────────────────────────────────────────────────┘
       │                                     │
       ▼                                     ▼
┌──────────────────┐              ┌────────────────────────────┐
│  Hugging Face    │              │  Hugging Face Space        │
│  Model Hub       │              │                            │
│                  │              │  ┌──────────────────────┐  │
│  sentiment-      │              │  │ 🔮 Inference Tab     │  │
│  distilbert      │◀─ ─ ─ ─ ─ ─│  │  Sentiment + Metrics │  │
│  (reference      │              │  └──────────────────────┘  │
│   stats + model) │              │  ┌──────────────────────┐  │
└──────────────────┘              │  │ 📊 Monitoring Tab    │  │
                                  │  │  Drift + Predictions │  │
                                  │  └──────────────────────┘  │
                                  └────────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

- [GitHub](https://github.com) account
- [Hugging Face](https://huggingface.co) account
- Hugging Face access token with `write` permissions

### 2. Create a Hugging Face Token

1. Go to [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → **Write** → give it a name like `ml-observability`
3. Copy the token

### 3. Fork & Deploy

**Option A: One-click** (recommended)

1. Clone this repo
2. Push to your own GitHub repo
3. Add your HF token as a repository secret:

```bash
gh secret set HF_TOKEN -R your-username/ml-observability-platform
```

Or add it manually in **Settings → Secrets and variables → Actions → New repository secret**:
- **Name:** `HF_TOKEN`
- **Value:** `hf_your_token_here`

4. Push to `main` — the CI/CD pipeline runs automatically:
   - Trains DistilBERT on IMDB (20k samples)
   - Pushes model to `DynamicKarabo/sentiment-distilbert` (or your namespace)
   - Deploys the Gradio app to HF Spaces
   - Verifies the Space is running

**Option B: Manual deploy**

```bash
# Train locally
pip install -e ".[train]"
python model/train.py

# Push to HF Hub
export HF_TOKEN=hf_your_token_here
python model/push_to_hub.py

# Deploy to HF Spaces
python scripts/deploy_space.py
```

## 🎯 Features

### 🔮 Inference Tab
- Real-time sentiment classification
- Confidence scores + probability breakdown
- Latency tracking per request
- Example inputs to test immediately

### 📊 Monitoring Tab
- **Drift Status** — 🟢 Normal / 🟠 Warning / 🔴 Critical
- **PSI (Population Stability Index)** — distribution shift metric
- **Confidence Shift** — mean confidence change over time
- **Label Shift** — POSITIVE/NEGATIVE ratio change
- **Confidence Scatter** — every prediction plotted over time
- **Label Distribution** — pie chart of POSITIVE vs NEGATIVE
- **Drift Gauge** — composite score from 0-100%
- **Recent Predictions** — last 20 predictions with latency

### 🔄 CI/CD Pipeline
- **On push**: Train → Evaluate → Push → Deploy → Verify
- **Weekly retrain**: Every Monday 6AM UTC — fresh model, updated drift baseline
- **Smart skip**: Training skips if only `app/` files change
- **Reference sync**: Drift detector always uses latest training stats

## 🧪 Drift Detection

The platform uses **3 signals** for drift detection:

| Signal | Method | Weight |
|--------|--------|--------|
| Confidence Distribution | PSI (Population Stability Index) | 40% |
| Confidence Shift | Mean confidence delta | 30% |
| Label Shift | POSITIVE/NEGATIVE ratio delta | 30% |

Thresholds:
- **Normal** (🟢): Score < 0.15 — all good
- **Mild** (🟡): Score 0.15–0.40 — watch but no action
- **Warning** (🟠): Score 0.40–0.70 — investigate
- **Critical** (🔴): Score > 0.70 — retrain recommended

## 📁 Project Structure

```
ml-observability-platform/
├── .github/workflows/
│   └── ci-cd.yml              # Full MLOps pipeline
├── app/
│   ├── app.py                  # Gradio app (inference + monitoring)
│   ├── requirements.txt        # Space dependencies
│   ├── drift_detector.py       # PSI + statistical drift detection
│   ├── prediction_store.py     # Prediction logging (JSONL)
│   └── model/                  # Reference stats (synced from training)
├── model/
│   ├── train.py                # DistilBERT fine-tuning script
│   ├── push_to_hub.py          # Upload model + reference stats
│   └── requirements.txt        # Training dependencies
├── scripts/
│   ├── deploy_space.py         # HF Space creation + upload
│   ├── sync_reference_stats.py # Sync drift baseline to app/
│   └── health_check.py         # Verify Space is running
├── tests/
│   └── test_drift.py           # Drift detection unit tests
├── data/                       # Prediction logs (auto-created)
├── pyproject.toml
└── README.md
```

## ⚙️ Configuration

Set these environment variables to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face write token (required) |
| `HF_MODEL_ID` | `DynamicKarabo/sentiment-distilbert` | Model repo on Hub |
| `HF_SPACE_ID` | `DynamicKarabo/ml-observability` | Space to deploy |

## 📊 Portfolio Impact

This project demonstrates every skill an ML Platform Engineer needs:

- ✅ **ML Pipeline Engineering** — automated training, evaluation, registry
- ✅ **ML Serving** — Gradio deployment with production-grade inference
- ✅ **Model Monitoring** — real-time drift detection with statistical methods
- ✅ **CI/CD for ML** — GitHub Actions pipeline with smart triggers
- ✅ **Infrastructure as Code** — reproducible deploys, zero manual steps
- ✅ **Full Lifecycle** — train → deploy → monitor → retrain loop

## 📝 License

MIT
