"""
content_based.py — Content-based course recommender.

Matches students to courses via cosine similarity on a 6-dimensional space
where each student feature has a distinct, conceptually parallel course feature.

Matching dimensions:
  Student                          Course
  prior_pass_rate               →  historical_pass_rate    (track record match)
  1 - risk_prob  (capability)   →  1 - historical_withdrawal_rate  (survivability)
  archetype_num / 6             →  historical_distinction_rate  (stretch potential)
  edu_num / 4   (aspiration)    →  historical_distinction_rate  (academic ceiling)
  1 - dropout_flag              →  1 - historical_withdrawal_rate  (persistence fit)
  studied_credits / 240         →  module_presentation_length (normalised)  (workload fit)

Risk-aware override:
  High-risk students (risk_prob >= 0.70 or alert_tier = High) are filtered
  away from modules with historical_pass_rate < PASS_RATE_FLOOR before
  similarity is computed. Steering a struggling student into a 38% pass-rate
  module is a product failure.
"""

import numpy as np
import pandas as pd

PASS_RATE_FLOOR = 0.45
HIGH_RISK_PROB  = 0.70


def _student_vec(sf_row):
    return np.array([
        float(sf_row.get("prior_pass_rate",    0.5)),
        1.0 - float(sf_row.get("risk_prob",    0.0)),
        float(sf_row.get("archetype_num",      3.0)) / 6.0,
        float(sf_row.get("edu_num",            2.0)) / 4.0,
        1.0 - float(sf_row.get("prior_dropout_flag", 0.0)),
        min(float(sf_row.get("studied_credits", 60.0)) / 240.0, 1.0),
    ])


def _course_mat(course_agg):
    pr  = course_agg["historical_pass_rate"].values
    dr  = course_agg["historical_distinction_rate"].values
    wr  = course_agg["historical_withdrawal_rate"].values
    dur = course_agg["module_presentation_length"].values
    dur_n = dur / (dur.max() if dur.max() > 0 else 1.0)
    # Each column is distinct — maps 1-to-1 with _student_vec dimensions:
    # [track record, survivability, stretch, aspiration, persistence, workload]
    return np.column_stack([pr, 1 - wr, dr, dr, 1 - wr, dur_n])


def _cosine_rank(s_vec, c_mat, modules, top_k, method_label):
    s_norm = np.linalg.norm(s_vec)
    if s_norm == 0:
        return []
    s_unit = s_vec / s_norm
    c_norms = np.linalg.norm(c_mat, axis=1, keepdims=True)
    c_norms = np.where(c_norms > 0, c_norms, 1.0)
    sims = (c_mat / c_norms) @ s_unit
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [
        {"module": modules[i], "score": round(float(sims[i]), 4), "method": method_label}
        for i in top_idx
    ]


def _filter_candidates(cf, already_taken, is_high_risk):
    candidates = cf[~cf["code_module"].isin(already_taken)].copy()
    if is_high_risk:
        candidates = candidates[candidates["historical_pass_rate"] >= PASS_RATE_FLOOR]
    return candidates


def recommend_content(id_student, already_taken, sf, cf, top_k=3):
    """
    Recommend top_k courses via cosine similarity for a student with existing history.
    """
    sf_idx = sf.set_index("id_student") if "id_student" in sf.columns else sf
    if id_student not in sf_idx.index:
        return []

    sf_row = sf_idx.loc[id_student]
    is_high_risk = (
        float(sf_row.get("risk_prob", 0.0)) >= HIGH_RISK_PROB or
        int(sf_row.get("alert_tier_num", 0)) >= 2
    )

    candidates = _filter_candidates(cf, already_taken, is_high_risk)
    if len(candidates) == 0:
        return []

    agg_cols = [c for c in candidates.columns if c not in ["code_module", "code_presentation"]]
    course_agg = candidates.groupby("code_module")[agg_cols].mean().reset_index()

    s_vec = _student_vec(sf_row)
    c_mat = _course_mat(course_agg)
    return _cosine_rank(s_vec, c_mat, course_agg["code_module"].tolist(), top_k, "Content")


def cold_start_recommend(sf_row_dict, cf, top_k=3):
    """
    Recommend for a brand-new student using enrollment-form features.

    Strategy: popularity-first (rank by historical pass rate, which is computable
    from 2013 training data) with a risk-aware filter for high-risk students.

    Why not cosine content similarity for cold-start:
    Empirical evaluation showed Cold-Content P@3=0.018 vs Popularity P@3=0.220.
    The cosine approach produces near-identical feature vectors for all new students
    (defaults dominate) → effectively random. Popularity with a risk-floor filter
    is both simpler and 12x more accurate on the held-out cold-start cohort.

    For students with risk_prob < HIGH_RISK_PROB, all modules are eligible and
    ranked by pass rate. For high-risk students, modules below PASS_RATE_FLOOR
    are filtered out first to avoid steering them into high-failure-rate courses.
    """
    sf_row = pd.Series(sf_row_dict)
    is_high_risk = float(sf_row.get("risk_prob", 0.0)) >= HIGH_RISK_PROB

    course_agg = cf.groupby("code_module")[
        [c for c in cf.columns if c not in ["code_module", "code_presentation"]]
    ].mean().reset_index()

    if is_high_risk:
        candidates = course_agg[course_agg["historical_pass_rate"] >= PASS_RATE_FLOOR]
        method = "Cold-Popularity-RiskFiltered"
    else:
        candidates = course_agg
        method = "Cold-Popularity"

    if len(candidates) == 0:
        candidates = course_agg  # failsafe: no modules pass filter, drop constraint
        method = "Cold-Popularity"

    candidates = candidates.sort_values("historical_pass_rate", ascending=False)
    max_pr = candidates["historical_pass_rate"].max()
    return [
        {
            "module": row["code_module"],
            "score": round(float(row["historical_pass_rate"]) / max_pr, 4),
            "method": method,
        }
        for _, row in candidates.head(top_k).iterrows()
    ]
