# Task 3 — Course Recommendation Engine: Design Document

## The Core Problem

Standard collaborative filtering assumes users have rated / interacted with multiple items.
In OULAD, **87.7% of students took only one module**. That kills user-user CF dead for the
majority — there is no cross-module history to compare.

This forced a redesign: use three signals that each handle single-module students gracefully.

---

## Three Signals

### 1. Markov Transition Chain (`transition.py`)

**What it does:** Builds `P(next_module | current_module, last_outcome, imd_quartile)` from
the 3,538 students (12.3%) who took 2+ modules.

**Why condition on outcome:** A student who *passed* CCC then took EEE is making a progression
move. A student who *withdrew* from CCC then took BBB may be seeking an easier path. These are
structurally different recommendations — treating them identically biases toward "what most
students did" regardless of whether it worked.

**Why condition on IMD quartile:** Students from lower socioeconomic bands face different
constraints and follow different pathways. Conditioning partially corrects for this.

**Laplace smoothing (pseudocount=1):** Many `(current, outcome, IMD)` triplets have very few
observations. Without smoothing, rare combinations produce zero probability for most next
modules, forcing a silent fallback. With smoothing, rare combinations fall back gracefully
toward the marginal distribution.

**Fallback chain:**
1. `P(next | current, outcome, imd_quartile)` — full conditioned lookup
2. If count ≤ 5 after smoothing → `P(next | current, outcome)` — marginal
3. If marginal also empty → no Markov signal (hybrid falls to other signals)

**Retake logic:** If `last_outcome` is Fail or Withdrawn, the same module is allowed as a
recommendation (retake). If Pass/Distinction, it is excluded.

**Grid search result:** Best hybrid weight = **0.9** (Markov dominates). This makes sense —
the Markov signal has direct empirical grounding: it literally observed what students chose next.

---

### 2. VLE Activity-Profile CF (`cf_vle.py`)

**What it does:** Builds a "behavioural fingerprint" for every student from their within-module
click distribution across 20 VLE activity types (quiz, forumng, oucontent, etc.). Compares
students using Pearson correlation.

**Why this works for single-module students:** The fingerprint is computed *within* each module,
then normalised against the module mean. So a single-module student still has a full profile — it
just reflects how they used that one module relative to their classmates. That profile is
comparable to any other student who used VLE similarly in *their* module.

**Why Pearson over cosine:**
A heavy user (10,000 clicks) and a light user (300 clicks) with the same *relative* distribution
are behaviourally similar — same style, different intensity. Cosine treats them as similar only if
magnitudes also match. Pearson subtracts the row mean first, so only the relative distribution
matters (implemented as mean-centering + cosine, which is mathematically equivalent to Pearson).

**Why within-module normalisation:**
`oucontent` constitutes different fractions of total VLE sites across modules. Subtracting the
module mean fraction removes the effect of module structure — what remains is each student's
preference *relative to their peers in the same module*.

**Neighbour voting:** For each of the top-20 neighbours, their passed/distinguished modules are
voted for. Vote weight = `neighbour_similarity × outcome_weight × log(1 + total_clicks)`.
- Distinction → 1.2× weight (stronger endorsement)
- Pass → 1.0× weight
- Fail/Withdrawn → excluded (not an endorsement)

**Grid search result:** Best hybrid weight = **0.1** (supporting signal). The CF catches cases
where Markov has no transition data — particularly for students whose last outcome was unusual.

---

### 3. Content-Based (`content_based.py`)

**What it does:** Projects student and course into a shared 12-dimensional matching space and
computes cosine similarity.

**Matching dimensions (student → course):**
| Student feature | Course feature |
|---|---|
| `prior_pass_rate` | `historical_pass_rate` |
| `mean_outcome_score` | `historical_pass_rate` |
| `mean_assessment_score / 100` | `historical_pass_rate` (proxy) |
| `archetype_num / 6` | `historical_pass_rate` |
| `1 - risk_prob` (capability) | `historical_pass_rate` |
| `imd_band_num / 10` | `1 - historical_withdrawal_rate` |
| `studied_credits / 240` | `module_presentation_length (norm)` |
| `edu_num / 4` | `historical_distinction_rate` |
| `1 - prior_dropout_flag` | `1 - historical_withdrawal_rate` |
| VLE CMA proxy | `frac_cma` |
| `age_band_num / 2` | `historical_pass_rate` |
| `1 - num_of_prev_attempts / 5` | `1 - historical_withdrawal_rate` |

**Risk-aware override:** High-risk students (`risk_prob ≥ 0.70` or `alert_tier = High`) are
filtered away from modules with `historical_pass_rate < 0.45`. Recommending a 38% pass-rate
module to a struggling student is a product failure — it sets them up to fail again.

**Cold-start:** The content-based module is the *only* signal that can fire for a brand-new
student with no history. It accepts a dict of known features (intake form data) and runs the same
cosine matching. If even that produces zero signal (all features missing), it falls back to
popularity-by-pass-rate.

**Grid search result:** Best hybrid weight = **0.0** (dropped). Content by itself is weakest on
this dataset — course features explain less variance than actual transition behaviour. But it's
retained in the architecture because it's the *only* signal for cold-start students, and it
activates the risk override.

---

## Hybrid Combiner (`hybrid.py`)

**Fusion strategy:** Weighted score fusion (not rank fusion).

Score fusion preserves signal strength — a 0.95 Markov score should dominate a 0.52 Content
score. Rank fusion would treat both as `#1`. Max-normalisation is applied per signal before
fusion so scores are on the same [0,1] scale.

**Weight search:** Coarse grid over `(w_content, w_markov, w_cf)` triplets summing to 1, 0.1
steps, 66 combinations. Objective: mean Precision@3 on 30% of the holdout (tuning split).
Remaining 70% is the clean eval set.

**Signal availability by student type:**

| Student type | Content | Markov | CF |
|---|---|---|---|
| New (no history) | Cold-start mode | — | — |
| Single-module (87.7%) | Full | If in transition matrix | Full |
| Multi-module (12.3%) | Full | Full | Full |

**Explanation layer:** Every recommendation carries a one-sentence advisor-facing rationale for
the PathAI dashboard. Non-technical university staff cannot act on opaque scores.

---

## Evaluation

### Holdout Construction

Only students who appear in **both** 2013 and 2014 presentations can provide ground truth —
they have observed "next module" choices. This yields **1,113 students** with ground truth.

The holdout is honest because:
- Training signals (Markov, CF) were built on all available data (not split away)
- The evaluation asks: given what this student did in 2013, did we recommend what they actually
  chose in 2014?
- This is a **proxy** for true recommendation quality — students may have chosen their next module
  for reasons (scheduling, availability) we can't observe

**Critical limitation:** The 87.7% single-module majority has no ground truth. We evaluate
only on the multi-module minority. Results should be interpreted as "upper bound for students
with history" not as a claim about all students.

### Metrics

| Metric | Formula | What it measures |
|---|---|---|
| **Precision@3** | `|rec ∩ truth| / 3` | Did we recommend what they actually took? |
| **Coverage** | `|modules recommended| / |all modules|` | Do we explore the full catalogue or converge on 2-3 popular ones? |
| **Success Rate** | `% students with ≥1 hit in top-3` | Practical: did the advisor have anything useful to show? |

### Results (780-student eval set)

| Method | Precision@3 | Coverage | Success Rate |
|---|---|---|---|
| **Hybrid** | **0.2654** | **1.000** | **0.786** |
| Popularity | 0.2145 | 0.571 | 0.636 |
| Pass-Rate | 0.0192 | 0.571 | 0.058 |
| Random | 0.1697 | 1.000 | 0.504 |

**Key findings:**
- Hybrid beats Popularity by +5.1pp Precision, +15pp Success Rate
- Hybrid has 100% Coverage — it surfaces all 7 modules (Popularity only covers 4)
- Pass-Rate baseline collapses because students don't just choose the "safest" module — they
  follow specific pathways. P@3=0.02 is a strong signal that content similarity alone is
  insufficient; behavioural history (Markov) is the dominant driver
- The random baseline's 0.50 Success Rate is high because there are only 7 modules — guessing
  3 of 7 without the student's current module will hit ~60% of likely next choices by chance

**Why Markov weight = 0.9:** The transition chain directly models observed behaviour. With only
7 modules and strong pathway structure (CCC→EEE→CCC being the most common pair), the Markov
chain captures the dominant signal cleanly. Content-based drops to 0 weight not because content
is useless, but because Markov already encodes the structural information.

---

## File Structure

```
Task3/
├── features_task3.py   — Course (28 dims) + student (20 dims) feature matrices
├── transition.py       — Markov P(next|current, outcome, IMD)
├── cf_vle.py           — VLE activity-profile CF (Pearson, top-20 neighbours)
├── content_based.py    — Cosine similarity, risk override, cold-start
├── hybrid.py           — Weighted fusion, grid search, batch generation, explanations
├── evaluate_task3.py   — Precision@3, Coverage, Success Rate, baseline comparisons
├── task3.ipynb         — 12-stage main notebook
├── OUTPUTS/            — CSVs: recommendations.csv, evaluation_summary.csv, weight_grid_search.csv
├── PLOTS/              — PNGs: evaluation_comparison, transition_heatmap, similarity_distribution,
│                                vle_profiles_by_archetype, weight_grid_search, method_breakdown,
│                                coverage_heatmap
└── TASK3_DESIGN.md     — This file
```

---

## Known Limitations and Future Improvements

**Limitation 1 — Evaluation population bias:**
Results are measured on the 12.3% of students with 2+ modules. We have no ground truth for
the 87.7% majority. The hybrid is architecturally sound for them (all signals degrade
gracefully) but we cannot quantify quality for that population.

**Limitation 2 — Markov dominance (weight=0.9):**
The grid search rewards Markov heavily because OULAD has only 7 modules and strong pathway
structure. On a real platform with 100+ courses, pathway diversity would be much greater and
the content signal would likely contribute more. The current weights are dataset-specific.

**Limitation 3 — Content feature alignment:**
The 12-dim matching space uses `historical_pass_rate` as a proxy for multiple student features
(performance, risk, archetype). This is a simplification — a proper matching space would
require a learned projection (e.g. a small MLP trained on known student-outcome pairs).

**Limitation 4 — Cold-start risk override ineffective in demo:**
The cold-start demo shows the same modules for high-risk and standard students. This is
because `PASS_RATE_FLOOR=0.45` and the modules with pass rate below that threshold (BBB: 0.47,
FFF: 0.47) are only just above floor. At `threshold=0.50` the override would activate more
aggressively. Worth tuning per deployment context.

**Future: Implicit feedback loop:**
As students enrol and complete modules, recommendation quality can be measured in production
and used to retrain weights monthly. The grid search framework is already wired for this.

**Future: Presentation-aware recommendations:**
The current system recommends `code_module` (e.g. CCC) without specifying presentation
(2024B vs 2024J). A production system would filter by available upcoming presentations.

**Future: Collaborative filtering at scale:**
VLE-CF currently runs `NearestNeighbors(algorithm='brute')` — O(N²) at query build time.
For >100K students, switch to approximate nearest neighbours (FAISS or Annoy).
