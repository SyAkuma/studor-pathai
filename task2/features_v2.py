"""
features_v2.py — Task 2v2 feature engineering.

Hard constraint: every feature must be derivable from data observable at or
before day 41 (week ≤ 6). No leakage.

Changes from v1:
  - Loads weekly_scores_v2.csv (Task1v2 output, behavioral-only score)
  - Drops mean_completion_w6 (redundant with direct assessment computation)
  - Adds any_submission_w6 binary flag (separates non-submitters from late submitters)
  - mean_timeliness_w6 = NaN for non-submitters, not 30 (different signal)
  - code_module one-hot encoded (not label-encoded ordinal)
  - Archetype snapshot from archetype_week at week 6 (real-time, not retrospective)
  - Includes n_archetype_changes_w6 and transition_direction_w6
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress

WEEK_LIMIT = 6   # hard cutoff — never change
GROUP = ["id_student", "code_module", "code_presentation"]


def _slope(series):
    s = series.dropna()
    if len(s) < 2:
        return 0.0
    x = np.arange(len(s))
    return float(linregress(x, s.values).slope)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir=".."):
    print("Loading CSVs...")
    weekly_scores = pd.read_csv(f"{data_dir}/task1/OUTPUTS/weekly_scores_v2.csv")
    st_info       = pd.read_csv(f"{data_dir}/studentInfo.csv")
    st_reg        = pd.read_csv(f"{data_dir}/studentRegistration.csv")
    st_ass        = pd.read_csv(f"{data_dir}/studentAssessment.csv")
    assessments   = pd.read_csv(f"{data_dir}/assessments.csv")
    print("Done.")
    return weekly_scores, st_info, st_reg, st_ass, assessments


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def build_labels(st_info):
    df = st_info[GROUP + ["final_result"]].copy()
    df["label"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    return df.drop(columns=["final_result"])


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------

def feat_score_aggregates(weekly_scores):
    """
    Aggregate Task1v2 engagement_score over weeks 0-6.
    Does NOT include mean_completion_w6 — redundant with Block 2 direct computation.
    """
    w6 = weekly_scores[weekly_scores["week"] <= WEEK_LIMIT].copy()

    agg = w6.groupby(GROUP)["engagement_score"].agg(
        mean_score_w6  ="mean",
        min_score_w6   ="min",
        std_score_w6   ="std",
        weeks_below_40 =lambda x: (x < 40).sum(),
        weeks_present  ="count",
    ).reset_index()

    slopes = (
        w6.sort_values(GROUP + ["week"])
        .groupby(GROUP)["engagement_score"]
        .apply(_slope)
        .reset_index(name="score_slope_w6")
    )

    last_week = (
        w6.sort_values(GROUP + ["week"])
        .groupby(GROUP)
        .last()
        .reset_index()[GROUP + ["engagement_score"]]
        .rename(columns={"engagement_score": "score_at_w6"})
    )

    # raw VLE aggregates (behavioral, not score-derived)
    vle_cols = {
        "weekly_clicks":      "total_clicks_w6",
        "days_active":        "mean_days_active_w6",
        "activity_diversity": "mean_activity_diversity_w6",
    }
    avail = {k: v for k, v in vle_cols.items() if k in w6.columns}
    if avail:
        vle_agg = w6.groupby(GROUP).agg(**{v: (k, "mean") for k, v in avail.items()}).reset_index()
        if "weekly_clicks" in w6.columns:
            click_sum = w6.groupby(GROUP)["weekly_clicks"].sum().reset_index(name="total_clicks_w6")
            vle_agg = vle_agg.drop(columns=["total_clicks_w6"], errors="ignore").merge(click_sum, on=GROUP, how="left")
    else:
        vle_agg = pd.DataFrame(columns=GROUP)

    df = agg.merge(slopes, on=GROUP, how="left").merge(last_week, on=GROUP, how="left")
    if len(vle_agg.columns) > len(GROUP):
        df = df.merge(vle_agg, on=GROUP, how="left")

    return df


def feat_assessment(st_ass, assessments):
    """
    Assessment features, week ≤ 6, non-exam only.

    Key v2 changes vs v1:
    - any_submission_w6 binary flag added (non-submitters = 0, not confused with late)
    - mean_timeliness_w6 = NaN for non-submitters (imputed with cohort median later),
      NOT filled with 30 — that conflated absence with extreme lateness
    - completion_rate_w6 = 0 for non-submitters (correct: they submitted nothing)
    """
    non_exam = assessments[assessments["assessment_type"] != "Exam"].copy()
    non_exam["week"] = non_exam["date"] // 7
    non_exam_w6 = non_exam[non_exam["week"] <= WEEK_LIMIT]

    sub = st_ass.merge(
        non_exam_w6[["id_assessment", "code_module", "code_presentation", "date", "week"]],
        on="id_assessment", how="inner"
    )
    sub["days_late"] = sub["date_submitted"] - sub["date"]

    sub_agg = sub.groupby(GROUP).agg(
        n_submitted        =("id_assessment", "count"),
        mean_score_ass_w6  =("score",         "mean"),
        mean_timeliness_w6 =("days_late",      "mean"),
    ).reset_index()
    sub_agg["any_submission_w6"] = 1

    due_counts = (
        non_exam_w6.groupby(["code_module", "code_presentation"])
        .size().reset_index(name="n_due")
    )

    df = sub_agg.merge(due_counts, on=["code_module", "code_presentation"], how="left")
    df["completion_rate_w6"] = (df["n_submitted"] / df["n_due"]).clip(0, 1)

    return df[GROUP + ["completion_rate_w6", "any_submission_w6",
                        "mean_score_ass_w6", "mean_timeliness_w6"]]


def feat_demographics(st_info):
    """Static features from studentInfo — all observable at enrollment."""
    df = st_info[GROUP + [
        "gender", "imd_band", "age_band",
        "num_of_prev_attempts", "studied_credits", "disability"
    ]].copy()

    imd_order = {
        "0-10%": 1, "10-20": 2, "20-30%": 3, "30-40%": 4, "40-50%": 5,
        "50-60%": 6, "60-70%": 7, "70-80%": 8, "80-90%": 9, "90-100%": 10
    }
    age_order = {"0-35": 0, "35-55": 1, "55<=": 2}

    df["imd_band_num"] = df["imd_band"].map(imd_order)
    df["age_band_num"] = df["age_band"].map(age_order)
    df["gender_F"]     = (df["gender"] == "F").astype(int)
    df["disability_Y"] = (df["disability"] == "Y").astype(int)

    return df[GROUP + ["imd_band_num", "age_band_num", "gender_F", "disability_Y",
                        "num_of_prev_attempts", "studied_credits"]]


def feat_registration(st_reg):
    """
    Registration timing. date_unregistration deliberately excluded — it is the outcome.
    """
    df = st_reg[GROUP + ["date_registration"]].copy()
    df["early_registrant"] = (df["date_registration"] < 0).astype(int)
    df["reg_days_before"]  = (-df["date_registration"]).clip(lower=0)
    return df[GROUP + ["early_registrant", "reg_days_before"]]


def feat_module_onehot(df):
    """
    One-hot encode code_module.
    v1 used label-encoding (ordinal) which implied AAA < BBB < CCC — meaningless.
    Modules are unordered categories; each gets its own binary column.
    """
    dummies = pd.get_dummies(df["code_module"], prefix="mod").astype(int)
    return pd.concat([df[GROUP], dummies], axis=1)


def feat_archetype_snapshot(weekly_scores):
    """
    Week-6 snapshot of real-time archetype (archetype_week column from Task1v2).
    Uses only backward-looking labels — no 'Late Recoverer', no retrospective types.
    Also derives:
      - n_archetype_changes_w6: how many times archetype switched in weeks 0-6
      - transition_direction_w6: direction of last transition (encoded numerically)
    """
    if "archetype_week" not in weekly_scores.columns:
        print("WARNING: archetype_week not found in weekly_scores_v2.csv — skipping archetype block")
        return pd.DataFrame(columns=GROUP)

    w6 = weekly_scores[weekly_scores["week"] <= WEEK_LIMIT].copy()

    # snapshot at last available week ≤ 6
    arch_snap = (
        w6.sort_values(GROUP + ["week"])
        .groupby(GROUP)
        .last()
        .reset_index()[GROUP + ["archetype_week"]]
        .rename(columns={"archetype_week": "archetype_at_w6"})
    )

    # number of archetype changes
    changes = (
        w6[w6["archetype_changed"].notna()]
        .groupby(GROUP)["archetype_changed"]
        .sum()
        .reset_index(name="n_archetype_changes_w6")
    ) if "archetype_changed" in w6.columns else pd.DataFrame(columns=GROUP + ["n_archetype_changes_w6"])

    # last transition direction
    dir_snap = (
        w6[w6["transition_direction"].notna() & (w6["transition_direction"] != "first_week")]
        .sort_values(GROUP + ["week"])
        .groupby(GROUP)
        .last()
        .reset_index()[GROUP + ["transition_direction"]]
        .rename(columns={"transition_direction": "transition_direction_w6"})
    ) if "transition_direction" in w6.columns else pd.DataFrame(columns=GROUP + ["transition_direction_w6"])

    df = arch_snap.merge(changes, on=GROUP, how="left").merge(dir_snap, on=GROUP, how="left")

    # one-hot archetype — only real-time valid labels
    VALID_RT = ["Ghost", "Early Dropout", "Struggling", "Coasting",
                "Steady Engager", "Recovering", "Anxious"]
    df["archetype_at_w6"] = df["archetype_at_w6"].where(
        df["archetype_at_w6"].isin(VALID_RT), other="Ghost"
    )
    dummies = pd.get_dummies(df["archetype_at_w6"], prefix="arch_w6").astype(int)
    for label in VALID_RT:
        col = f"arch_w6_{label.replace(' ', '_')}"
        if col not in dummies.columns:
            dummies[col] = 0

    # encode transition direction numerically
    dir_map = {"improving": 1, "stable": 0, "declining": -1}
    df["transition_dir_num"] = df["transition_direction_w6"].map(dir_map).fillna(0)

    result = pd.concat([df[GROUP], dummies, df[["n_archetype_changes_w6", "transition_dir_num"]]], axis=1)
    return result


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_feature_matrix(data_dir=".."):
    """
    Returns full DataFrame with features + label column.
    Columns GROUP + feature_cols + ['label', 'code_presentation'].
    code_presentation kept for temporal split.
    """
    weekly_scores, st_info, st_reg, st_ass, assessments = load_data(data_dir)

    labels   = build_labels(st_info)
    score_f  = feat_score_aggregates(weekly_scores)
    assess_f = feat_assessment(st_ass, assessments)
    demo_f   = feat_demographics(st_info)
    reg_f    = feat_registration(st_reg)
    mod_f    = feat_module_onehot(st_info[GROUP].drop_duplicates())
    arch_f   = feat_archetype_snapshot(weekly_scores)

    df = labels.copy()
    for feat_df in [score_f, assess_f, demo_f, reg_f, mod_f, arch_f]:
        if len(feat_df.columns) > len(GROUP):
            # drop any columns that already exist in df (except GROUP keys)
            overlap = [c for c in feat_df.columns if c in df.columns and c not in GROUP]
            feat_df = feat_df.drop(columns=overlap)
            df = df.merge(feat_df, on=GROUP, how="left")

    # ── Imputation strategy ────────────────────────────────────────────────
    # VLE behavioral: no activity = 0 engagement (student was simply absent)
    vle_zero_cols = [
        "mean_score_w6", "min_score_w6", "std_score_w6", "weeks_below_40",
        "weeks_present", "score_slope_w6", "score_at_w6",
        "total_clicks_w6", "mean_days_active_w6", "mean_activity_diversity_w6",
    ]
    for col in vle_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Assessment: non-submitters = 0 submitted, 0 completion, unknown timeliness
    df["completion_rate_w6"]  = df["completion_rate_w6"].fillna(0)
    df["any_submission_w6"]   = df["any_submission_w6"].fillna(0)
    df["mean_score_ass_w6"]   = df["mean_score_ass_w6"].fillna(0)
    # mean_timeliness_w6: NaN for non-submitters → impute with cohort median
    # (not 30 — absence ≠ extreme lateness)
    df["mean_timeliness_w6"]  = df["mean_timeliness_w6"].fillna(df["mean_timeliness_w6"].median())

    # Demographics: fill with median
    for col in ["imd_band_num", "age_band_num", "reg_days_before"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Archetype: missing student (no VLE) → ghost-like, all zeros
    arch_cols = [c for c in df.columns if c.startswith("arch_w6_") or c in ["n_archetype_changes_w6", "transition_dir_num"]]
    df[arch_cols] = df[arch_cols].fillna(0)

    # Final safety net
    df = df.fillna(0)

    feature_cols = [c for c in df.columns if c not in GROUP + ["label"]]
    print(f"Feature matrix: {len(df):,} students, {len(feature_cols)} features")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"\nFeature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  {c}")

    return df[GROUP + feature_cols + ["label"]]
