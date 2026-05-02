"""
features_task3.py — Course and student feature matrices for Task 3.

Two outputs:
  course_features.csv  — one row per (code_module, code_presentation), ~28 features
  student_features.csv — one row per unique id_student, ~20 features

Design rationale:
  Course features describe the *character* of a module: what kinds of VLE activities
  it uses (learning style), how intensive the assessments are, and how hard it is
  historically. These are stable properties of the course, not student-specific.

  Student features describe who the student is: demographics, prior performance
  (aggregated across all modules they have taken), engagement archetype from Task 1,
  and risk profile from Task 2 (where available — 2014 cohort only).

  Both vectors are normalised before similarity computation. They are NOT in the same
  feature space — the content-based recommender bridges them by projecting into a shared
  matching space (see content_based.py).
"""

import numpy as np
import pandas as pd

DATA_DIR = ".."
PRES_ORDER = {"2013B": 0, "2013J": 1, "2014B": 2, "2014J": 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _imd_encode(s):
    mapping = {
        "0-10%": 1, "10-20": 2, "20-30%": 3, "30-40%": 4, "40-50%": 5,
        "50-60%": 6, "60-70%": 7, "70-80%": 8, "80-90%": 9, "90-100%": 10
    }
    return pd.to_numeric(s.map(mapping), errors="coerce")


def _age_encode(s):
    return s.map({"0-35": 0, "35-55": 1, "55<=": 2})


def _edu_encode(s):
    order = {
        "No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
        "HE Qualification": 3, "Post Graduate Qualification": 4
    }
    return s.map(order).fillna(2)


# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------

def load_raw(data_dir=DATA_DIR):
    print("Loading raw CSVs...")
    si       = pd.read_csv(f"{data_dir}/studentInfo.csv")
    vle_meta = pd.read_csv(f"{data_dir}/vle.csv")
    ass      = pd.read_csv(f"{data_dir}/assessments.csv")
    st_ass   = pd.read_csv(f"{data_dir}/studentAssessment.csv")
    courses  = pd.read_csv(f"{data_dir}/courses.csv")
    arch     = pd.read_csv(f"{data_dir}/Task1v2/OUTPUTS/student_archetypes_v2.csv")
    alerts   = pd.read_csv(f"{data_dir}/Task2v2/OUTPUTS/student_alerts_v2.csv")
    print("Done.")
    return si, vle_meta, ass, st_ass, courses, arch, alerts


# ---------------------------------------------------------------------------
# Course feature matrix
# ---------------------------------------------------------------------------

def build_course_features(vle_meta, ass, courses, si, data_dir=DATA_DIR):
    """
    One row per (code_module, code_presentation).

    Feature groups:
      1. VLE activity-type distribution (20 dims) — the learning style of the module
      2. Assessment structure (4 dims) — intensity, timing, type mix
      3. Historical difficulty (3 dims) — pass/distinction/withdrawal rates
      4. Duration (1 dim)
    """
    GROUP = ["code_module", "code_presentation"]

    # ── 1. VLE activity-type distribution ────────────────────────────────────
    # Fraction of VLE sites that are each activity type.
    # Rationale: characterises whether a module is discussion-heavy (high forumng),
    # assessment-heavy (high quiz), content-delivery (high oucontent/resource), etc.
    act_counts = (
        vle_meta.groupby(GROUP + ["activity_type"])
        .size().unstack(fill_value=0)
        .reset_index()
    )
    # normalise to fractions (each row sums to 1 over activity types)
    act_cols = [c for c in act_counts.columns if c not in GROUP]
    act_counts[act_cols] = act_counts[act_cols].div(
        act_counts[act_cols].sum(axis=1).replace(0, 1), axis=0
    )
    act_counts.columns = [
        f"act_{c}" if c not in GROUP else c for c in act_counts.columns
    ]

    # ── 2. Assessment structure ───────────────────────────────────────────────
    # assessment_type: TMA (tutor-marked), CMA (computer-marked), Exam
    # Exam excluded from structural features — we care about formative assessment style.
    non_exam = ass[ass["assessment_type"] != "Exam"].copy()
    non_exam["week"] = non_exam["date"] // 7

    ass_struct = non_exam.groupby(GROUP).agg(
        n_assessments     =("id_assessment", "count"),
        frac_cma          =("assessment_type", lambda x: (x == "CMA").mean()),
        mean_weight       =("weight", "mean"),
        n_ass_by_w6       =("week", lambda x: (x <= 6).sum()),
    ).reset_index()

    # fraction of assessments due in first 6 weeks (early workload)
    ass_struct["frac_ass_early"] = (
        ass_struct["n_ass_by_w6"] / ass_struct["n_assessments"].replace(0, 1)
    ).clip(0, 1)

    # ── 3. Historical difficulty ──────────────────────────────────────────────
    # Computed from studentInfo — what fraction of enrolled students passed, etc.
    # Rationale: pass rates range 37.8% (CCC) to 71.0% (AAA) — enormous spread.
    # An at-risk student should not be recommended a 38% pass-rate module.
    hist = si.groupby(GROUP)["final_result"].agg(
        historical_pass_rate        =lambda x: x.isin(["Pass", "Distinction"]).mean(),
        historical_distinction_rate =lambda x: (x == "Distinction").mean(),
        historical_withdrawal_rate  =lambda x: (x == "Withdrawn").mean(),
    ).reset_index()

    # ── 4. Duration ──────────────────────────────────────────────────────────
    dur = courses[GROUP + ["module_presentation_length"]].copy()

    # ── Merge all course features ─────────────────────────────────────────────
    cf = (
        hist
        .merge(act_counts,  on=GROUP, how="left")
        .merge(ass_struct,  on=GROUP, how="left")
        .merge(dur,         on=GROUP, how="left")
    )
    cf = cf.fillna(0)
    print(f"Course features: {len(cf)} rows × {len(cf.columns)} columns")
    return cf


# ---------------------------------------------------------------------------
# Student feature matrix
# ---------------------------------------------------------------------------

def build_student_features(si, st_ass, ass, arch, alerts):
    """
    One row per unique id_student (aggregated across all their module enrollments).

    Feature groups:
      1. Demographics (6 dims) — stable, from enrollment
      2. Performance history (4 dims) — aggregated across all prior modules
      3. Engagement archetype (1 dim, encoded) — from Task 1
      4. Risk profile (2 dims) — from Task 2 (2014 cohort; 0 elsewhere)
      5. Module history metadata (2 dims) — n_modules_taken, last_module

    Representation choice: one row per id_student means we collapse multiple
    module enrollments. For a student who took BBB then CCC, we aggregate their
    performance across both. Their "current" module is their most recent one
    (by presentation order).
    """
    # ── 1. Demographics ───────────────────────────────────────────────────────
    # Take one row per student — demographics are constant across their rows
    demo = (
        si.sort_values("code_presentation")
        .groupby("id_student")
        .last()  # most recent enrollment's demographics
        .reset_index()[["id_student", "gender", "imd_band", "age_band",
                        "num_of_prev_attempts", "studied_credits",
                        "disability", "highest_education"]]
    )
    demo["imd_band_num"]  = _imd_encode(demo["imd_band"])
    demo["age_band_num"]  = _age_encode(demo["age_band"])
    demo["gender_F"]      = (demo["gender"] == "F").astype(int)
    demo["disability_Y"]  = (demo["disability"] == "Y").astype(int)
    demo["edu_num"]       = _edu_encode(demo["highest_education"])

    demo = demo[["id_student", "imd_band_num", "age_band_num", "gender_F",
                 "disability_Y", "num_of_prev_attempts", "studied_credits", "edu_num"]]

    # ── 2. Performance history ────────────────────────────────────────────────
    # Aggregate across all module enrollments per student.
    outcome_map = {"Distinction": 1.2, "Pass": 1.0, "Fail": 0.3, "Withdrawn": 0.0}
    si_perf = si.copy()
    si_perf["outcome_score"] = si_perf["final_result"].map(outcome_map)
    si_perf["passed"]        = si_perf["final_result"].isin(["Pass", "Distinction"]).astype(int)
    si_perf["withdrawn"]     = (si_perf["final_result"] == "Withdrawn").astype(int)

    perf = si_perf.groupby("id_student").agg(
        n_modules_taken   =("code_module",    "count"),
        prior_pass_rate   =("passed",          "mean"),
        prior_dropout_flag=("withdrawn",       "max"),   # 1 if ever withdrew
        mean_outcome_score=("outcome_score",   "mean"),
    ).reset_index()

    # Last module and outcome (most recent by presentation order)
    si_sorted = (
        si.copy()
        .assign(pres_ord=lambda d: d["code_presentation"].map(PRES_ORDER).fillna(0))
        .sort_values(["id_student", "pres_ord"])
    )
    last_mod = (
        si_sorted.groupby("id_student")
        .last()
        .reset_index()[["id_student", "code_module", "code_presentation", "final_result"]]
        .rename(columns={"code_module": "last_module",
                         "code_presentation": "last_presentation",
                         "final_result": "last_outcome"})
    )

    # ── 3. Assessment scores ──────────────────────────────────────────────────
    # Mean score across all submitted non-exam assessments
    non_exam_ids = ass[ass["assessment_type"] != "Exam"]["id_assessment"]
    valid_sub = st_ass[
        st_ass["id_assessment"].isin(non_exam_ids) & st_ass["score"].notna()
    ]
    mean_scores = (
        valid_sub.groupby("id_student")["score"]
        .mean()
        .reset_index(name="mean_assessment_score")
    )

    # ── 4. Engagement archetype ───────────────────────────────────────────────
    # Most recent archetype per student
    arch_enc = {"Steady Engager": 6, "Recovering": 5, "Coasting": 4,
                "Anxious": 3, "Struggling": 2, "Early Dropout": 1, "Ghost": 0,
                "Late Recoverer": 5, "Burnt-out Achiever": 3}
    arch_last = (
        arch.assign(pres_ord=lambda d: d["code_presentation"].map(PRES_ORDER).fillna(0))
        .sort_values(["id_student", "pres_ord"])
        .groupby("id_student")
        .last()
        .reset_index()[["id_student", "archetype_final", "mean_score"]]
        .rename(columns={"mean_score": "arch_mean_score"})
    )
    arch_last["archetype_num"] = arch_last["archetype_final"].map(arch_enc).fillna(3)

    # ── 5. Risk profile ───────────────────────────────────────────────────────
    # From Task 2 alerts (2014 cohort only — others get 0)
    risk = (
        alerts.assign(pres_ord=lambda d: d["code_presentation"].map(PRES_ORDER).fillna(0))
        .sort_values(["id_student", "pres_ord"])
        .groupby("id_student")
        .last()
        .reset_index()[["id_student", "risk_prob", "alert_tier"]]
    )
    tier_map = {"High": 2, "Medium": 1, "Watch": 0}
    risk["alert_tier_num"] = risk["alert_tier"].map(tier_map).fillna(0)
    risk = risk[["id_student", "risk_prob", "alert_tier_num"]]

    # ── Merge all student features ────────────────────────────────────────────
    sf = (
        demo
        .merge(perf,        on="id_student", how="left")
        .merge(last_mod,    on="id_student", how="left")
        .merge(mean_scores, on="id_student", how="left")
        .merge(arch_last[["id_student", "archetype_num", "arch_mean_score"]],
               on="id_student", how="left")
        .merge(risk,        on="id_student", how="left")
    )

    # Impute
    sf["n_modules_taken"]    = sf["n_modules_taken"].fillna(1)
    sf["prior_pass_rate"]    = sf["prior_pass_rate"].fillna(0.5)
    sf["prior_dropout_flag"] = sf["prior_dropout_flag"].fillna(0)
    sf["mean_outcome_score"] = sf["mean_outcome_score"].fillna(0.5)
    sf["mean_assessment_score"] = sf["mean_assessment_score"].fillna(50.0)
    sf["archetype_num"]      = sf["archetype_num"].fillna(3)
    sf["arch_mean_score"]    = sf["arch_mean_score"].fillna(50.0)
    sf["risk_prob"]          = sf["risk_prob"].fillna(0.0)
    sf["alert_tier_num"]     = sf["alert_tier_num"].fillna(0)

    print(f"Student features: {len(sf)} rows × {len(sf.columns)} columns")
    return sf


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_all(data_dir=DATA_DIR):
    si, vle_meta, ass, st_ass, courses, arch, alerts = load_raw(data_dir)
    cf = build_course_features(vle_meta, ass, courses, si, data_dir)
    sf = build_student_features(si, st_ass, ass, arch, alerts)
    return cf, sf, si, vle_meta
