# Studor PathAI Engine — DS Screening Project

Intelligent early-warning and course recommendation layer for university students, built on the **OULAD (Open University Learning Analytics Dataset)**.

---

## Project Structure

```
studor-pathai/
├── task1/          Behavioral engagement scoring framework (35 pts)
├── task2/          Predictive disengagement model (35 pts)
├── task3/          Course recommendation engine (30 pts)
├── requirements.txt
└── README.md
```

Data files (7 CSVs) are **not included in the repo** due to size. See the Data Setup section below.

---

## Tasks

### Task 1 — Behavioral Scoring Framework
- 17 behavioral features engineered from VLE clickstream across 5 domains (Temporal, Assessment, Content, Trajectory, Cross-domain)
- Dynamic 0–100 weekly engagement score per student, recomputed each week
- 9 student archetypes (Steady Engager, Ghost, Coasting, Willing but Struggling, etc.)
- Evaluation: AUROC per week, archetype separation, score calibration, feature ablation, early warning timeliness

**Entry point:** `task1/task1v2.ipynb`

### Task 2 — Predictive Disengagement Model
- Binary classifier: Withdrawn/Fail = 1, Pass/Distinction = 0
- Hard constraint: features from Week ≤ 6 only — no leakage
- Models: Logistic Regression, Random Forest, XGBoost (winner)
- Optimised for Recall via F2 threshold sweep + Platt calibration
- SHAP explanations per student; staff alert table with top-3 risk reasons

**Results (XGBoost, threshold=0.16):** AUROC 0.861 · Recall 0.977 · Precision 0.575 · F2 0.857

**Entry point:** `task2/task2v2.ipynb`

### Task 3 — Course Recommendation Engine
- Content-based: course metadata + student profile cosine similarity
- Collaborative filtering: VLE interaction patterns via k-NN on click vectors
- Markov chain: module transition probabilities conditioned on prior outcome
- Hybrid fusion with grid-searched weights
- Cold-start: popularity + pass-rate floor for new students with no history
- Evaluation: Precision@3 on 2014 holdout vs Popularity, PassRate, Random baselines

**Entry point:** `task3/task3.ipynb`

---

## Setup

### 1. Clone the repo
```bash
git clone <repo-url>
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
Download from [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset) and place all 7 CSVs in the **repo root**:

```
assessments.csv
courses.csv
studentAssessment.csv
studentInfo.csv
studentRegistration.csv
studentVle.csv   (10M+ rows — ~500MB)
vle.csv
```

### 5. Run notebooks in order

Task 1 must run before Task 2 (generates `weekly_scores_v2.csv` and `student_archetypes_v2.csv`).
Task 2 must run before Task 3 (generates `student_alerts_v2.csv`).

```bash
# From repo root
jupyter lab
```

Open and run:
1. `task1/task1v2.ipynb`
2. `task2/task2v2.ipynb`
3. `task3/task3.ipynb`

Pre-executed versions with all outputs are included as `*_executed.ipynb` in each task folder.

---

## Key Output Files

| File | Description |
|------|-------------|
| `task1/OUTPUTS/weekly_scores_v2.csv` | Engagement score per student per week |
| `task1/OUTPUTS/student_archetypes_v2.csv` | Final archetype label per student |
| `task2/OUTPUTS/student_alerts_v2.csv` | Risk probability + top-3 SHAP reasons per student |
| `task3/OUTPUTS/recommendations.csv` | Top-3 course recommendations per student |
| `task3/OUTPUTS/evaluation_summary.csv` | P@3 comparison across all methods |

---

## Known Limitations

See `CRITICAL_REVIEW.md` for a full independent audit. Key items:

- Task 1: Activity diversity uses `nunique()` rather than Shannon entropy; survivorship bias in late-week percentile thresholds
- Task 2: Platt calibration and threshold selection share the same validation set (optimistic Brier score)
- Task 3: Markov transition matrix trained on full data including 2014 evaluation period (P@3 slightly inflated); cold-start content vector defaults are identical for all new students

---

## Requirements

Python 3.10+. All dependencies in `requirements.txt`. Notable packages: `xgboost`, `shap`, `optuna`, `scikit-learn >= 1.3`.
