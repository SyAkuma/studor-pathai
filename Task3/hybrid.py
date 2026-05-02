"""
hybrid.py — Hybrid recommender combining Content, Markov, and VLE-CF signals.

Fusion: weighted score fusion after per-signal max-normalisation.
  Score fusion preserves signal strength — a 0.95 Markov score should
  dominate a 0.52 Content score. Rank fusion would treat both as equal (#1).

Signal availability:
  New student (no history)    → Content (cold-start) only
  Single-module student       → Content + VLE-CF + Markov (if in transition table)
  Multi-module student        → Content + VLE-CF + Markov

Weights are tuned by grid search on a held-out tuning split (see evaluate_task3.py).
"""

import itertools
import numpy as np
import pandas as pd

from content_based import recommend_content
from transition   import recommend_markov
from cf_vle       import recommend_cf, build_si_index

DEFAULT_W = {"content": 0.15, "markov": 0.60, "cf": 0.25}


def _score_dict(recs):
    return {r["module"]: r["score"] for r in recs}


def _max_norm(d):
    if not d:
        return d
    mx = max(d.values())
    return {k: v / mx for k, v in d.items()} if mx > 0 else d


def hybrid_recommend(
    id_student, already_taken, sf, cf,
    neighbours, si, weights,
    trans_df, marg_df,
    weights_hybrid=None,
    top_k=3,
):
    """
    Produce top_k hybrid recommendations for one student.

    Returns list of dicts:
      {module, hybrid_score, content_score, markov_score, cf_score,
       method_used, explanation}
    """
    if weights_hybrid is None:
        weights_hybrid = DEFAULT_W
    w_c = weights_hybrid.get("content", 0.15)
    w_m = weights_hybrid.get("markov",  0.60)
    w_f = weights_hybrid.get("cf",      0.25)

    sf_idx = sf.set_index("id_student") if "id_student" in sf.columns else sf

    # ── Pull signals ──────────────────────────────────────────────────────────
    c_recs = recommend_content(id_student, already_taken, sf, cf, top_k=10)
    si_index = weights_hybrid.get("_si_index") if weights_hybrid else None
    f_recs = recommend_cf(id_student, already_taken, neighbours, si, weights, top_k=10, si_index=si_index)

    m_recs = []
    if id_student in sf_idx.index:
        row          = sf_idx.loc[id_student]
        last_module  = row.get("last_module")
        last_outcome = row.get("last_outcome")
        if pd.notna(last_module) and pd.notna(last_outcome):
            m_recs = recommend_markov(
                last_module, last_outcome,
                already_taken, trans_df, marg_df, top_k=10
            )

    c_scores = _max_norm(_score_dict(c_recs))
    m_scores = _max_norm(_score_dict(m_recs))
    f_scores = _max_norm(_score_dict(f_recs))

    all_modules = set(c_scores) | set(m_scores) | set(f_scores)

    rows = []
    for mod in all_modules:
        cs = c_scores.get(mod, 0.0)
        ms = m_scores.get(mod, 0.0)
        fs = f_scores.get(mod, 0.0)
        active = (["Content"] if cs > 0 else []) + \
                 (["Markov"]  if ms > 0 else []) + \
                 (["CF"]      if fs > 0 else [])
        hybrid = w_c * cs + w_m * ms + w_f * fs
        rows.append({
            "module":        mod,
            "hybrid_score":  round(hybrid, 4),
            "content_score": round(cs, 4),
            "markov_score":  round(ms, 4),
            "cf_score":      round(fs, 4),
            "method_used":   "+".join(active) if active else "None",
        })

    rows.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_rows = rows[:top_k]
    for r in top_rows:
        r["explanation"] = _explain(r)
    return top_rows


def _explain(row):
    parts = []
    if "Content"  in row["method_used"] and row["content_score"] > 0.5:
        parts.append("matches your academic profile")
    if "Markov"   in row["method_used"] and row["markov_score"]  > 0.3:
        parts.append("commonly chosen next by students with similar outcomes")
    if "CF"       in row["method_used"] and row["cf_score"]      > 0.3:
        parts.append("popular among students with similar study patterns")
    if not parts:
        parts.append("recommended based on available signals")
    return f"{row['module']}: " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Grid-search weight tuning
# ---------------------------------------------------------------------------

def grid_search_weights(
    tune_students, already_taken_map, sf, cf,
    neighbours, si, weights,
    trans_df, marg_df,
    ground_truth_map,
    top_k=3, step=0.1,
):
    """
    Grid search (w_content, w_markov, w_cf) summing to 1.
    Objective: mean Precision@k on tune_students.
    Returns best_weights dict and full results DataFrame.
    """
    vals = [round(v * step, 2) for v in range(0, int(1 / step) + 1)]
    triplets = [
        (c, m, f) for c, m, f in itertools.product(vals, repeat=3)
        if abs(c + m + f - 1.0) < 1e-9
    ]

    si_index = build_si_index(si)
    results = []
    for c, m, f in triplets:
        w = {"content": c, "markov": m, "cf": f, "_si_index": si_index}
        prec_list = []
        for sid in tune_students:
            taken = already_taken_map.get(sid, set())
            recs  = hybrid_recommend(
                sid, taken, sf, cf, neighbours, si, weights,
                trans_df, marg_df, weights_hybrid=w, top_k=top_k
            )
            gt = ground_truth_map.get(sid, set())
            if gt:
                prec_list.append(len({r["module"] for r in recs} & gt) / top_k)
        mean_p = float(np.mean(prec_list)) if prec_list else 0.0
        results.append({"w_content": c, "w_markov": m, "w_cf": f,
                        "precision_at_k": round(mean_p, 4)})

    results_df = pd.DataFrame(results).sort_values("precision_at_k", ascending=False)
    best = results_df.iloc[0]
    best_weights = {"content": best["w_content"],
                    "markov":  best["w_markov"],
                    "cf":      best["w_cf"]}
    print(f"Best weights: {best_weights}  =>  Precision@{top_k} = {best['precision_at_k']:.4f}")
    return best_weights, results_df


# ---------------------------------------------------------------------------
# Batch recommendation
# ---------------------------------------------------------------------------

def recommend_all(
    student_ids, already_taken_map, sf, cf,
    neighbours, si, weights,
    trans_df, marg_df,
    weights_hybrid=None, top_k=3,
):
    si_index = build_si_index(si)
    w = dict(weights_hybrid) if weights_hybrid else dict(DEFAULT_W)
    w["_si_index"] = si_index

    all_rows = []
    for sid in student_ids:
        taken = already_taken_map.get(sid, set())
        recs  = hybrid_recommend(
            sid, taken, sf, cf, neighbours, si, weights,
            trans_df, marg_df, weights_hybrid=w, top_k=top_k
        )
        for rank, r in enumerate(recs, 1):
            all_rows.append({"id_student": sid, "rank": rank, **r})
    return pd.DataFrame(all_rows)
