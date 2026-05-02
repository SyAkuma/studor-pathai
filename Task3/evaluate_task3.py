"""
evaluate_task3.py — Evaluation framework for Task 3 recommender.

Two evaluation populations:

  Main holdout (returning students):
    Students who appeared in 2013 AND 2014. Their 2013 modules = already_taken.
    Ground truth = new modules they enrolled in during 2014.
    1,113 students, 780 used for eval (30% held for weight tuning).

  Cold-start holdout (brand-new students):
    Students who appear ONLY in 2014 and took both 2014B and 2014J.
    Their 2014B enrollment = first module (simulates new student).
    Ground truth = what they enrolled in for 2014J.
    633 evaluable students.
    Only cold_start_recommend() fires — no Markov, no VLE-CF.

Metrics:
  Precision@K  : fraction of top-K recs that the student actually enrolled in
  Coverage     : fraction of catalogue appearing in at least one recommendation
  Success Rate : fraction of students with ≥1 hit in top-K

Baselines:
  Popularity — top-K by overall enrollment count
  Random     — uniform sample from available pool
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

PRES_ORDER = {"2013B": 0, "2013J": 1, "2014B": 2, "2014J": 3}


# ---------------------------------------------------------------------------
# Holdout construction
# ---------------------------------------------------------------------------

def build_holdout(si):
    """
    Returns:
      holdout_students   : list of id_student who have 2013 history AND 2014 enrollment
      already_taken_map  : {id_student: set of modules from 2013 (training) period}
      ground_truth_map   : {id_student: set of modules first seen in 2014 (new modules)}
    """
    si = si.copy()
    si["pres_ord"] = si["code_presentation"].map(PRES_ORDER).fillna(0)
    si["is_2013"]  = si["code_presentation"].isin(["2013B", "2013J"])
    si["is_2014"]  = si["code_presentation"].isin(["2014B", "2014J"])

    train_mods = si[si["is_2013"]].groupby("id_student")["code_module"].apply(set).rename("train_mods")
    test_mods  = si[si["is_2014"]].groupby("id_student")["code_module"].apply(set).rename("test_mods")

    merged = pd.concat([train_mods, test_mods], axis=1).dropna()
    # new modules = in 2014 but not in 2013
    merged["new_mods"] = merged.apply(
        lambda r: r["test_mods"] - r["train_mods"], axis=1
    )
    merged = merged[merged["new_mods"].apply(len) > 0]

    holdout_students  = merged.index.tolist()
    already_taken_map = merged["train_mods"].to_dict()
    ground_truth_map  = merged["new_mods"].to_dict()

    n = len(holdout_students)
    print(f"Holdout: {n:,} students with observed next-module ground truth")
    return holdout_students, already_taken_map, ground_truth_map


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(recs_map, ground_truth_map, k=3):
    """
    recs_map : {id_student: list of {module, ...}} or {id_student: set of module codes}
    Returns mean Precision@K across all students with ground truth.
    """
    scores = []
    for sid, gt in ground_truth_map.items():
        if sid not in recs_map:
            continue
        recs = recs_map[sid]
        if isinstance(recs, (list, )):
            rec_mods = {r["module"] if isinstance(r, dict) else r for r in recs[:k]}
        else:
            rec_mods = set(list(recs)[:k])
        scores.append(len(rec_mods & gt) / k)
    return float(np.mean(scores)) if scores else 0.0


def coverage(recs_map, all_modules):
    """Fraction of modules that appear in at least one recommendation."""
    recommended = set()
    for recs in recs_map.values():
        for r in recs:
            mod = r["module"] if isinstance(r, dict) else r
            recommended.add(mod)
    return len(recommended) / len(all_modules) if all_modules else 0.0


def success_rate(recs_map, ground_truth_map, k=3):
    """Fraction of students who got ≥1 correct recommendation in top-K."""
    hits = 0
    total = 0
    for sid, gt in ground_truth_map.items():
        if sid not in recs_map:
            continue
        total += 1
        recs = recs_map[sid]
        rec_mods = {r["module"] if isinstance(r, dict) else r for r in recs[:k]}
        if rec_mods & gt:
            hits += 1
    return hits / total if total else 0.0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def popularity_baseline(holdout_students, already_taken_map, si, top_k=3):
    """Recommend top-K most popular modules, excluding already taken."""
    pop = si["code_module"].value_counts()
    recs_map = {}
    for sid in holdout_students:
        taken = already_taken_map.get(sid, set())
        recs = []
        for mod, _ in pop.items():
            if mod not in taken:
                recs.append(mod)
            if len(recs) == top_k:
                break
        recs_map[sid] = [{"module": m} for m in recs]
    return recs_map


def pass_rate_baseline(holdout_students, already_taken_map, cf, top_k=3):
    """Recommend top-K modules by historical pass rate, excluding already taken."""
    pr = (
        cf.groupby("code_module")["historical_pass_rate"]
        .mean()
        .sort_values(ascending=False)
    )
    recs_map = {}
    for sid in holdout_students:
        taken = already_taken_map.get(sid, set())
        recs = [{"module": m} for m in pr.index if m not in taken][:top_k]
        recs_map[sid] = recs
    return recs_map


def random_baseline(holdout_students, already_taken_map, all_modules, top_k=3, seed=42):
    """Randomly sample top_k modules, excluding already taken."""
    rng = np.random.default_rng(seed)
    recs_map = {}
    all_mods = list(all_modules)
    for sid in holdout_students:
        taken = already_taken_map.get(sid, set())
        pool  = [m for m in all_mods if m not in taken]
        chosen = rng.choice(pool, size=min(top_k, len(pool)), replace=False)
        recs_map[sid] = [{"module": m} for m in chosen]
    return recs_map


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(
    hybrid_recs_df,
    holdout_students, already_taken_map, ground_truth_map,
    si, cf,
    all_modules=None,
    top_k=3,
    output_dir="OUTPUTS",
    plots_dir="PLOTS",
):
    """
    Run full evaluation: hybrid vs three baselines.
    hybrid_recs_df : output of recommend_all(), columns [id_student, rank, module, ...]

    Returns summary DataFrame.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if all_modules is None:
        all_modules = set(si["code_module"].unique())

    # Build recs maps
    hybrid_map = (
        hybrid_recs_df.groupby("id_student")
        .apply(lambda g: g.sort_values("rank")[["module"]].to_dict("records"))
        .to_dict()
    )
    pop_map  = popularity_baseline(holdout_students, already_taken_map, si, top_k)
    pr_map   = pass_rate_baseline(holdout_students, already_taken_map, cf, top_k)
    rand_map = random_baseline(holdout_students, already_taken_map, all_modules, top_k)

    rows = []
    for name, rm in [("Hybrid", hybrid_map), ("Popularity", pop_map),
                     ("PassRate", pr_map), ("Random", rand_map)]:
        p = precision_at_k(rm, ground_truth_map, k=top_k)
        c = coverage(rm, all_modules)
        s = success_rate(rm, ground_truth_map, k=top_k)
        rows.append({"Method": name, f"Precision@{top_k}": round(p, 4),
                     "Coverage": round(c, 4), "SuccessRate": round(s, 4)})
        print(f"{name:12s}  P@{top_k}={p:.4f}  Coverage={c:.4f}  Success={s:.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{output_dir}/evaluation_summary.csv", index=False)

    # Plot precision comparison
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [f"Precision@{top_k}", "Coverage", "SuccessRate"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, metric in zip(axes, metrics):
        ax.bar(summary["Method"], summary[metric], color=colors)
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        for i, v in enumerate(summary[metric]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.suptitle("Task 3 — Recommender Evaluation", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/evaluation_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {plots_dir}/evaluation_comparison.png")

    # Module coverage heatmap
    _plot_coverage_heatmap(hybrid_map, holdout_students, all_modules, plots_dir)

    return summary


def _plot_coverage_heatmap(hybrid_map, holdout_students, all_modules, output_dir):
    """How often each module is recommended — bar chart."""
    from collections import Counter
    counts = Counter()
    for sid in holdout_students:
        for r in hybrid_map.get(sid, []):
            counts[r["module"] if isinstance(r, dict) else r] += 1

    mods = sorted(all_modules)
    vals = [counts.get(m, 0) for m in mods]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(mods, vals, color="#4C72B0")
    ax.set_xlabel("Module")
    ax.set_ylabel("Times Recommended")
    ax.set_title("Recommendation Coverage — How Often Each Module Appears")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/coverage_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/coverage_heatmap.png")


# ---------------------------------------------------------------------------
# Method breakdown
# ---------------------------------------------------------------------------

def method_breakdown(hybrid_recs_df, output_dir="OUTPUTS", plots_dir="PLOTS"):
    """
    How often does each signal contribute (Content / Markov / CF)?
    """
    if "method_used" not in hybrid_recs_df.columns:
        return

    from collections import Counter
    c = Counter()
    for m in hybrid_recs_df["method_used"]:
        for part in str(m).split("+"):
            c[part.strip()] += 1
    total = len(hybrid_recs_df)

    print("\nMethod contribution breakdown:")
    for method, cnt in c.most_common():
        print(f"  {method:12s}: {cnt:,} ({cnt/total*100:.1f}%)")

    import os
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = list(c.keys())
    vals    = [c[m] for m in methods]
    ax.bar(methods, vals, color=["#4C72B0","#DD8452","#55A868","#999"])
    ax.set_title("Signal Contribution to Recommendations")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/method_breakdown.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Cold-start holdout
# ---------------------------------------------------------------------------

def build_coldstart_holdout(si):
    """
    Students who appear ONLY in 2014 and took both 2014B and 2014J.
      already_taken_map : {id_student: {module from 2014B}}
      ground_truth_map  : {id_student: {new modules from 2014J}}

    2014B simulates enrolment (their very first module).
    2014J is what they chose next — our ground truth.
    """
    s2013 = set(si[si["code_presentation"].isin(["2013B", "2013J"])]["id_student"])
    new_students = si[~si["id_student"].isin(s2013)].copy()

    b_mods = (
        new_students[new_students["code_presentation"] == "2014B"]
        .groupby("id_student")["code_module"].apply(set)
    )
    j_mods = (
        new_students[new_students["code_presentation"] == "2014J"]
        .groupby("id_student")["code_module"].apply(set)
    )
    both = pd.concat([b_mods.rename("b"), j_mods.rename("j")], axis=1).dropna()
    both["new"] = both.apply(lambda r: r["j"] - r["b"], axis=1)
    both = both[both["new"].apply(len) > 0]

    print(f"Cold-start holdout: {len(both):,} brand-new students with observed next-module")
    return (
        both.index.tolist(),
        both["b"].to_dict(),
        both["new"].to_dict(),
    )


def evaluate_coldstart(cf, si, coldstart_students, already_taken_map, ground_truth_map,
                       top_k=3, output_dir="OUTPUTS", plots_dir="PLOTS"):
    """
    Evaluate cold_start_recommend() vs popularity baseline on brand-new students.
    Uses only enrollment-form features extracted from studentInfo (no VLE, no history).
    """
    import os
    from content_based import cold_start_recommend

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)

    def imd_num(band):
        m = {"0-10%":1,"10-20":2,"20-30%":3,"30-40%":4,"40-50%":5,
             "50-60%":6,"60-70%":7,"70-80%":8,"80-90%":9,"90-100%":10}
        return float(m.get(band, 5))

    def age_num(a):
        return {"0-35": 0.0, "35-55": 1.0, "55<=": 2.0}.get(a, 0.0)

    def edu_num(e):
        order = {"No Formal quals":0,"Lower Than A Level":1,"A Level or Equivalent":2,
                 "HE Qualification":3,"Post Graduate Qualification":4}
        return float(order.get(e, 2))

    si_idx = si.sort_values("code_presentation").groupby("id_student").last()

    content_prec, pop_prec = [], []
    for sid in coldstart_students:
        gt   = ground_truth_map.get(sid, set())
        if not gt:
            continue
        row = si_idx.loc[sid] if sid in si_idx.index else {}

        features = {
            "imd_band_num":        imd_num(row.get("imd_band", "")),
            "age_band_num":        age_num(row.get("age_band", "")),
            "edu_num":             edu_num(row.get("highest_education", "")),
            "studied_credits":     float(row.get("studied_credits", 60)),
            "num_of_prev_attempts":float(row.get("num_of_prev_attempts", 0)),
            "prior_pass_rate":     0.5,
            "risk_prob":           0.0,
            "archetype_num":       3.0,
            "prior_dropout_flag":  0.0,
        }
        taken = already_taken_map.get(sid, set())

        # Content-based cold start
        recs_c = cold_start_recommend(features, cf, top_k=top_k)
        rec_mods_c = {r["module"] for r in recs_c}
        content_prec.append(len(rec_mods_c & gt) / top_k)

        # Popularity baseline
        pop = si["code_module"].value_counts()
        recs_p = [m for m in pop.index if m not in taken][:top_k]
        pop_prec.append(len(set(recs_p) & gt) / top_k)

    p_content = float(np.mean(content_prec)) if content_prec else 0.0
    p_pop     = float(np.mean(pop_prec))     if pop_prec     else 0.0
    sr_content = float(np.mean([v > 0 for v in content_prec])) if content_prec else 0.0
    sr_pop     = float(np.mean([v > 0 for v in pop_prec]))     if pop_prec     else 0.0

    print(f"\nCold-start evaluation ({len(content_prec):,} students):")
    print(f"  Content cold-start:  P@{top_k}={p_content:.4f}  Success={sr_content:.4f}")
    print(f"  Popularity baseline: P@{top_k}={p_pop:.4f}  Success={sr_pop:.4f}")

    summary = pd.DataFrame([
        {"Method": "Cold-Content",  f"Precision@{top_k}": round(p_content, 4), "SuccessRate": round(sr_content, 4)},
        {"Method": "Popularity",    f"Precision@{top_k}": round(p_pop, 4),     "SuccessRate": round(sr_pop, 4)},
    ])
    summary.to_csv(f"{output_dir}/coldstart_evaluation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, col in zip(axes, [f"Precision@{top_k}", "SuccessRate"]):
        ax.bar(summary["Method"], summary[col], color=["#4C72B0", "#DD8452"])
        ax.set_title(col)
        ax.set_ylim(0, 1)
        for i, v in enumerate(summary[col]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    plt.suptitle("Cold-Start Evaluation — Brand New 2014 Students", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/coldstart_evaluation.png", dpi=150)
    plt.close()
    print(f"Saved {plots_dir}/coldstart_evaluation.png")

    return summary
