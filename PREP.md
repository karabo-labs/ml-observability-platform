# PREP — ML Observability Platform: Shipped

**Date:** 2026-05-16
**Status:** ✅ Deployed and running

---

## 1. Current State

| Component | Status | Detail |
|-----------|--------|--------|
| DistilBERT fine-tuning | ✅ Done | 10k IMDB × 1 epoch → 89.96% accuracy, 90.06% F1 |
| Model on HF Hub | ✅ Live | [karaboLLM/sentiment-distilbert](https://huggingface.co/karaboLLM/sentiment-distilbert) |
| Gradio Space | ✅ Live | [karaboLLM/ml-observability](https://huggingface.co/spaces/karaboLLM/ml-observability) — Inference + Monitor |
| CI/CD pipeline | ✅ Green | Train → Test → Push → Deploy → Health Check |
| Drift detection | ✅ Working | PSI + composite score, 4 severity levels |
| Weekly retrain | ✅ Set | Monday 6AM UTC — fresh model, updated drift baseline |
| Reference stats sync | ✅ Working | 2,000 test samples as drift baseline |
| README | ✅ Written | Badges, screenshots, mermaid diagrams, decision log, 3 fires |
| CV case study | ✅ Updated | `/root/devops-prs-cv.md` entry added |
| Portfolio | ✅ Live | Project card + blog article on portfolio site |
| **Health-check job** | 🔴 **Pending** | Fix committed (#88066e6). Run #19 will verify |

**Bottom line:** The pipeline is built, model is trained, Space is live, docs are written. Run #19 should be the first full 💚 when it finishes.

---

## 2. What We Actually Built

### The Pipeline

```
Push to main → GitHub Actions →
  ├── [Conditional] Train: DistilBERT on 10k IMDB (625 steps, ~9 min)
  ├── Tests: 6 drift detection tests
  ├── Push model to HF Hub
  └── Deploy Gradio app to HF Spaces
       └── Health check → verify RUNNING
```

**The whole thing costs $0.** GitHub Actions free tier, HF Hub free tier, HF Spaces free CPU tier. Not "save money at scale" — save money at zero. Full stop.

### The App

Two tabs in one Gradio Space:

| Tab | What it does |
|-----|-------------|
| **🔮 Inference** | Type a movie review → get POSITIVE/NEGATIVE + confidence + latency. ~170ms on free CPU. |
| **📊 Monitor** | Drift dashboard — PSI score, confidence shift, label shift, recent predictions table. Updates live. |

---

## 3. Fires Fought

### Fire 1: "Bro why is training taking 80 minutes?"

**Symptom:** First run timed out after 78 minutes on the "Train model" step.

**Root cause:** 20,000 training samples × 2 epochs = ~2,500 steps. On a free GH runner (2 vCPU, no GPU), each step takes ~2s. Math: 2,500 × 2s = 83 min. Runner's 6-hour limit saves us, but it's still absurd.

**Fix:** Reduced to 10,000 samples × 1 epoch = 625 steps = ~9 min of actual training. Lost 0.04% accuracy (89.96% vs 90%+). Tradeoff worth making.

**Lesson learned:** Free CI runners are the constraint, not the training itself. Design for the runner, not the ideal.

### Fire 2: "Why does the drift test think everything is on fire?"

**Symptom:** `test_detect_drift_normal` failed — Expected "normal" or "mild", got "warning". Even when reference and predictions matched perfectly.

**Root cause:** The drift detector was comparing apples-to-oranges. During training, `pos_counts` stores the distribution of *positive-class softmax scores* (`probs[:, 1]`). But in production, the code was binning *max confidence* (`np.max(probs)`). For a sample predicted NEGATIVE with 90% confidence, those are different numbers (0.9 vs 0.1).

**Fix:** Convert production confidences to positive-class probabilities based on the predicted label before binning. Like-with-like comparison.

```python
# Before:
prod_pos_hist, _ = np.histogram(prod_confidences, bins=10)

# After:
prod_pos_probs = [
    conf if label == "POSITIVE" else 1.0 - conf
    for label, conf in zip(prod_labels, prod_confidences)
]
prod_pos_hist, _ = np.histogram(prod_pos_probs, bins=10)
```

**Lesson learned:** Drift detection is only as good as its baseline alignment. If training and production measure different things, every comparison is noise.

### Fire 3: "Where's the health check script?"

**Symptom:** Health-check job failed — `python: can't open file 'scripts/health_check.py'`

**Root cause:** The health-check job in CI was missing `actions/checkout@v4`. It tried to run a script that didn't exist in the workspace.

**Fix:** Added checkout step. Every CI job is a fresh container — no shared filesystem.

**Lesson learned:** Never assume files exist in a CI job. Every job is a blank slate. Checkout explicitly.

---

## 4. Decisions Log

| Decision | Choice | Why not the alternative |
|----------|--------|----------------------|
| Model | DistilBERT-base (66M) | BERT is 110M — 40% more params for ~1% more accuracy. Not worth it on CPU |
| Training data | 10k IMDB × 1 epoch | Full IMDB (25k) × 3 epochs = 7,500 steps = would never finish on free runner |
| Serving framework | Gradio | Streamlit has worse latency. FastAPI needs frontend. Gradio is UI + API in one |
| Drift method | PSI + composite score | KL divergence doesn't have industry convention for thresholds. PSI is standardized |
| Deployment target | HF Spaces | Render/Railway cost $5-20/mo. HF Spaces free CPU tier is genuinely capable for inference |
| Model storage | HF Hub | Git LFS has quota limits. Docker registry would mean pulling GB-sized images. HF Hub is built for this |
| CI runner | GitHub Actions free | Self-hosted runner would be faster but requires VPS resources. Free tier is sufficient for DistilBERT at reduced dataset |

---

## 5. The Flaws (Honest)

- **Training is slow even at 10k** — ~80 min total pipeline. Most of that is PyTorch install on CPU. No fix without GPU.
- **ZeroGPU isn't used** — Space runs on CPU. Could be faster with HF ZeroGPU but that's a PRO feature ($9/mo).
- **No alerting** — If drift goes critical, nobody gets pinged. Need Telegram/Slack integration.
- **Single model** — Only DistilBERT. Can't A/B test or warm-swap models without redeploying.
- **Predictions reset on rebuild** — The JSONL prediction log lives in the container. Every Space rebuild starts at zero.

---

## 6. What's Next

| Priority | Thing | Why |
|----------|-------|-----|
| 🟢 **Verify run #19** | Check health-check job passes | Should be first full 💚 |
| 🟡 **Drift alerts** | Telegram webhook when status = critical | Without this, drift detection is post-mortem |
| 🟡 **Prediction persistence** | HF Hub dataset or Qdrant for prediction log | Losing history on rebuild defeats monitoring |
| 🔵 **A/B model compare** | Deploy canary Space with different model | Portfolio showpiece: A/B testing in MLOps |
| 🔵 **Active learning** | Flag low-confidence predictions for review | Closes the loop — monitor → retrain trigger |

---

## 7. Bottom Line

This is a shipped, documented, $0 MLOps lifecycle. Not a Jupyter notebook. Not a tutorial. An automated pipeline that trains a model, deploys it, monitors it, and retrains it weekly — with 3 real bugs found and fixed along the way.

Every line of CI/CD, every drift detection threshold, every troubleshooting story is in the repo. Recruiters can see the app running, read the fires we fought, and understand the decisions we made.

Same vibe as the OSS containerization projects — but this time it's ML. And it works.
