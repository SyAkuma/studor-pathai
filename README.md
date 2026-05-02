# Studor PathAI Engine — DS Screening Project

Intelligent early-warning and course recommendation layer for university students, built on the **OULAD (Open University Learning Analytics Dataset)** — 32,593 students across 7 modules with clickstream, assessment, demographic, and outcome data.

---

## Project Structure

```
studor-pathai/
├── task1/                  Behavioral engagement scoring framework (35 pts)
│   ├── features_v2.py      10-feature engineering across 4 behavioral domains
│   ├── scoring_v2.py       0-100 weekly score + dual archetype system
│   ├── task1v2.ipynb       Main notebook (run this)
│   ├── OUTPUTS/            weekly_scores_v2.csv, student_archetypes_v2.csv
│   └── PLOTS/              8 diagnostic plots
├── task2/                  Predictive disengagement model (35 pts)
│   ├── features_v2.py      Week ≤ 6 feature aggregation, no leakage
│   ├── model_v2.py         XGBoost + Optuna + Platt calibration + SHAP alerts
│   ├── task2v2.ipynb       Main notebook (run this)
│   ├── task2v2_executed.ipynb  Pre-executed with all outputs
│   ├── OUTPUTS/            student_alerts_v2.csv
│   └── PLOTS/              9 diagnostic plots
├── Task3/                  Course recommendation engine (30 pts)
│   ├── features_task3.py   Student + course feature construction
│   ├── content_based.py    Content-based recommender
│   ├── cf_vle.py           Collaborative filtering (k-NN on VLE click vectors)
│   ├── transition.py       Markov chain on module transitions
│   ├── hybrid.py           Hybrid fusion + cold-start + explanations
│   ├── evaluate_task3.py   Precision@3, coverage, cold-start evaluation
│   ├── task3.ipynb         Main notebook (run this)
│   ├── task3_executed.ipynb  Pre-executed with all outputs
│   ├── OUTPUTS/            recommendations.csv, evaluation_summary.csv
│   └── PLOTS/              8 diagnostic plots
├── docs/                   Report, video script, understanding guide
├── requirements.txt
└── README.md
```

Data files (7 CSVs) are **not included** due to size — see Data Setup below.

---

## Results Summary

| Task | Key metric | Result |
|------|-----------|--------|
| Task 1 — Engagement Score | AUROC (score vs outcome, week 6) | ~0.78 across modules |
| Task 2 — Disengagement Model | AUROC / Recall / Precision | 0.849 / 0.777 / 0.675 |
| Task 3 — Recommendations | Precision@3 vs Popularity baseline | 0.259 vs 0.215 |

---

## Tasks

### Task 1 — Behavioral Scoring Framework
- **10 scored features** across 4 purely behavioral domains: VLE Presence, VLE Quality, Trajectory, Submission Behavior
- Dynamic 0–100 weekly engagement score per student, recomputed each week — not a static label
- Dual archetype system: 7 real-time labels (usable at week 6) + 2 retrospective labels (full-semester view)
- Submission domain weighted highest (30%) — confirmed by Task 2's SHAP analysis as the strongest behavioral predictor

**Entry point:** `task1/task1v2.ipynb`

### Task 2 — Predictive Disengagement Model
- Binary classifier: Withdrawn/Fail = 1, Pass/Distinction = 0
- **Hard constraint:** all features from Week ≤ 6 only — no leakage (`date_unregistration` explicitly excluded)
- Temporal train/test split: 2013 presentations → train, 2014 → test (reflects real deployment)
- XGBoost with 80-trial Optuna search, Platt calibration, F2-optimised threshold
- SHAP explanations per student; archetype-conditioned staff alerts with plain-English reasons

**Top 3 SHAP features:** assessment score by week 6 (0.976) · completion rate (0.735) · module identity (0.518)

**Entry point:** `task2/task2v2.ipynb` — pre-executed: `task2/task2v2_executed.ipynb`

### Task 3 — Course Recommendation Engine
- Three signals: content-based (cosine similarity), collaborative filtering (VLE k-NN), Markov chain (conditioned on prior outcome)
- Hybrid fusion with grid-searched weights — best: Markov 0.9, CF 0.1, Content 0.0
- Cold-start: popularity ranking filtered by withdrawal rate for high-risk students (from Task 2)
- Evaluated via Precision@3 on 2014 holdout against Popularity, PassRate, and Random baselines

**Entry point:** `task3/task3.ipynb` — pre-executed: `Task3/task3_executed.ipynb`

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/SyAkuma/studor-pathai.git
cd studor-pathai
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the OULAD dataset
Download from [Kaggle — OULAD dataset](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad) and place all 7 CSVs in the **repo root**:

```
assessments.csv
courses.csv
studentAssessment.csv
studentInfo.csv
studentRegistration.csv
studentVle.csv        (~500MB, 10M+ rows)
vle.csv
```

### 5. Run notebooks in order

**Execution order is strict** — each task depends on the previous task's outputs.

```bash
jupyter lab
```

1. `task1/task1v2.ipynb` → generates `task1/OUTPUTS/weekly_scores_v2.csv` and `student_archetypes_v2.csv`
2. `task2/task2v2.ipynb` → generates `task2/OUTPUTS/student_alerts_v2.csv`
3. `Task3/task3.ipynb`   → generates `Task3/OUTPUTS/recommendations.csv`

Pre-executed notebooks with all outputs visible are included (`task2v2_executed.ipynb`, `task3_executed.ipynb`). Task 1 outputs are regenerated on each run due to file size.

---

## Key Output Files

| File | Description |
|------|-------------|
| `task1/OUTPUTS/weekly_scores_v2.csv` | Engagement score per student per week (regenerated on run) |
| `task1/OUTPUTS/student_archetypes_v2.csv` | Real-time + retrospective archetype per student |
| `task2/OUTPUTS/student_alerts_v2.csv` | Risk probability, tier, top-3 SHAP reasons, recommended action |
| `Task3/OUTPUTS/recommendations.csv` | Top-3 module recommendations per student with explanations |
| `Task3/OUTPUTS/evaluation_summary.csv` | Precision@3 across all methods and baselines |

---

## Known Limitations

- **Task 1:** Activity diversity uses `nunique()` not Shannon entropy; survivorship bias in late-week cohort percentile thresholds; momentum feature undefined (zeroed) for students returning from zero-activity weeks
- **Task 2:** Platt calibration and threshold selection share the same validation fold — Brier score is slightly optimistic; test flag rate (53.6%) exceeded the 50% target ceiling
- **Task 3:** Markov chain trained on all presentations including 2014, which partially overlaps the evaluation holdout — P@3 is modestly inflated; cold-start defaults are identical for all new students

---

## Requirements

Python 3.10+. All dependencies in `requirements.txt`.

Key packages: `pandas >= 2.0`, `scikit-learn >= 1.3`, `xgboost >= 1.7`, `shap >= 0.42`, `optuna >= 3.0`, `scipy >= 1.10`, `jupyterlab >= 4.0`
