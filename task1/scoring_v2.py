"""
scoring_v2.py — Task 1v2: Weekly Engagement Score + Dual Archetype System

Score weights (sum = 1.0) — purely behavioral, no assessment score levels:
  VLE Presence  (25%): weekly_clicks 0.10, days_active 0.10, login_consistency 0.05
  VLE Quality   (25%): activity_diversity 0.08, deep_content_ratio 0.10, active_clicks_ratio 0.07
  Trajectory    (20%): trend_slope 0.12, engagement_momentum 0.08
  Submission    (30%): completion_rate 0.18, submission_timeliness 0.12

Archetype system — TWO outputs:
  A) archetype_week  — real-time, computed from data up to that week only.
     7 labels valid at any point in time (no future observation required).
     Used in: trajectory plots, transition tracking, Task 2v2 features (week-6 snapshot).

  B) archetype_final — retrospective, uses full course trajectory.
     Adds Late Recoverer and Burnt-out Achiever (require seeing the full semester).
     Used in: visualization, reporting only. NEVER feed into Task 2.
"""

import numpy as np
import pandas as pd

GROUP = ["id_student", "code_module", "code_presentation", "week"]

WEIGHTS = {
    # VLE Presence (25%)
    "weekly_clicks":       0.10,
    "days_active":         0.10,
    "login_consistency":   0.05,
    # VLE Quality (25%)
    "activity_diversity":  0.08,
    "deep_content_ratio":  0.10,
    "active_clicks_ratio": 0.07,
    # Trajectory (20%)
    "trend_slope":         0.12,
    "engagement_momentum": 0.08,
    # Submission Behavior (30%) — submission is behavior, not outcome
    "completion_rate":     0.18,
    "submission_timeliness": 0.12,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# Features where LOWER raw value = BETTER engagement
INVERT = {"submission_timeliness"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cohort_median_fill(df, col):
    """Fill NaN with cohort-week median (module × presentation × week)."""
    return df.groupby(["code_module", "code_presentation", "week"])[col].transform(
        lambda x: x.fillna(x.median())
    )


def _cohort_minmax(df, col):
    """Min-max normalise within cohort-week."""
    def _mm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / (rng if rng > 0 else 1.0)
    return df.groupby(["code_module", "code_presentation", "week"])[col].transform(_mm)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_scores(feature_df):
    """
    Input : wide feature table from features_v2.build_feature_table()
    Output: same DataFrame + <feature>_norm columns + engagement_score (0–100)
    """
    df    = feature_df.copy()
    score = pd.Series(0.0, index=df.index)

    for col, weight in WEIGHTS.items():
        if col not in df.columns:
            continue

        # 1. Impute NaN with cohort-week median (norm cols separately)
        df[col] = _cohort_median_fill(df, col)

        # 2. Invert where low raw value = better engagement
        working_col = col
        if col in INVERT:
            inv_col     = f"{col}_inv"
            df[inv_col] = -df[col]
            working_col = inv_col

        # 3. Normalise within cohort-week
        norm_col     = f"{col}_norm"
        df[norm_col] = _cohort_minmax(df, working_col).fillna(0)

        # 4. Accumulate weighted score
        score += weight * df[norm_col]

    df["engagement_score"] = (score * 100).clip(0, 100).round(1)
    return df


# ---------------------------------------------------------------------------
# A) Week-by-week real-time archetypes
# ---------------------------------------------------------------------------
#
# Rules use cohort percentiles computed from peers in the SAME week,
# so thresholds are always relative to "what is normal right now."
#
# Priority order (first match wins):
#   Ghost > Early Dropout > Recovering > Struggling > Anxious > Coasting > Steady Engager

def classify_archetypes_weekly(scored_df):
    """
    Assigns archetype_week to every (student, week) row using only information
    available up to and including that week.

    Fully vectorised — uses np.select with priority-ordered conditions (no row-wise apply).
    Returns scored_df with added columns: archetype_week, score_prev.
    """
    df = scored_df.sort_values(GROUP).copy()
    cw = ["code_module", "code_presentation", "week"]
    sk = ["id_student",  "code_module", "code_presentation"]

    def _pct(col, q, fill=0.0):
        return df.groupby(cw)[col].transform(
            lambda x: x.quantile(q) if x.notna().sum() > 0 else fill
        )

    sp25 = _pct("engagement_score",    0.25)
    sp50 = _pct("engagement_score",    0.50)
    cp15 = _pct("weekly_clicks",       0.15, fill=0.0)
    cp50 = _pct("weekly_clicks",       0.50, fill=0.0)
    hp75 = _pct("help_seeking_clicks", 0.75, fill=0.0)

    df["score_prev"] = df.groupby(sk)["engagement_score"].shift(1)

    score  = df["engagement_score"]
    clicks = df["weekly_clicks"].fillna(0)
    compl  = df["completion_rate"].fillna(0)
    slope  = df["trend_slope"].fillna(0)
    help_  = df["help_seeking_clicks"].fillna(0)
    prev   = df["score_prev"]

    c_ghost    = clicks <= cp15
    c_dropout  = ~c_ghost & (score <= sp25) & (slope < -0.5)
    c_recover  = ~c_ghost & ~c_dropout & prev.notna() & (prev < sp50) & (score >= sp50)
    c_struggle = ~c_ghost & ~c_dropout & ~c_recover & (score <= sp25) & (compl < 0.5)
    c_anxious  = (~c_ghost & ~c_dropout & ~c_recover & ~c_struggle &
                  (score > sp50) & (help_ > hp75) & (hp75 > 0) & (help_ > 5))
    c_coast    = (~c_ghost & ~c_dropout & ~c_recover & ~c_struggle & ~c_anxious &
                  (score > sp50) & (clicks < cp50))

    df["archetype_week"] = np.select(
        [c_ghost, c_dropout, c_recover, c_struggle, c_anxious, c_coast],
        ["Ghost", "Early Dropout", "Recovering", "Struggling", "Anxious", "Coasting"],
        default="Steady Engager",
    )
    return df


# ---------------------------------------------------------------------------
# Transition tracking
# ---------------------------------------------------------------------------

IMPROVING = {
    ("Ghost",          "Recovering"),
    ("Ghost",          "Struggling"),
    ("Ghost",          "Steady Engager"),
    ("Ghost",          "Coasting"),
    ("Early Dropout",  "Recovering"),
    ("Early Dropout",  "Struggling"),
    ("Early Dropout",  "Steady Engager"),
    ("Struggling",     "Recovering"),
    ("Struggling",     "Coasting"),
    ("Struggling",     "Steady Engager"),
    ("Coasting",       "Steady Engager"),
    ("Anxious",        "Steady Engager"),
}

DECLINING = {
    ("Steady Engager", "Coasting"),
    ("Steady Engager", "Struggling"),
    ("Steady Engager", "Early Dropout"),
    ("Steady Engager", "Ghost"),
    ("Recovering",     "Struggling"),
    ("Recovering",     "Early Dropout"),
    ("Recovering",     "Ghost"),
    ("Coasting",       "Struggling"),
    ("Coasting",       "Early Dropout"),
    ("Coasting",       "Ghost"),
    ("Anxious",        "Struggling"),
    ("Anxious",        "Ghost"),
    ("Anxious",        "Early Dropout"),
}


def compute_transitions(df_with_archetypes):
    """
    Adds per-student-week transition metadata:
      prev_archetype       — archetype in the prior week
      archetype_changed    — 1 if different from last week, 0 otherwise
      transition_direction — improving / declining / stable / first_week

    Fully vectorised — no row-wise apply.
    """
    df = df_with_archetypes.sort_values(GROUP).copy()
    sk = ["id_student", "code_module", "code_presentation"]

    df["prev_archetype"] = df.groupby(sk)["archetype_week"].shift(1)
    df["archetype_changed"] = (
        (df["archetype_week"] != df["prev_archetype"]) & df["prev_archetype"].notna()
    ).astype(int)

    # Build (prev, curr) tuple column as string key for set lookup
    prev_fill = df["prev_archetype"].fillna("")
    pairs = list(zip(prev_fill, df["archetype_week"]))

    is_first    = df["prev_archetype"].isna()
    is_improve  = pd.Series([p in IMPROVING for p in pairs], index=df.index) & ~is_first
    is_decline  = pd.Series([p in DECLINING  for p in pairs], index=df.index) & ~is_first

    df["transition_direction"] = np.select(
        [is_first, is_improve, is_decline],
        ["first_week", "improving", "declining"],
        default="stable",
    )
    return df




# ---------------------------------------------------------------------------
# B) End-of-semester retrospective archetypes
# ---------------------------------------------------------------------------

def classify_archetypes_final(scored_df):
    """
    Uses the FULL course trajectory to assign a summary archetype.
    Adds Late Recoverer and Burnt-out Achiever — these require seeing the complete semester.

    Output: one row per (id_student, code_module, code_presentation) with archetype_final.
    IMPORTANT: Never pass these labels to Task 2 as features.
    """
    df  = scored_df.sort_values(GROUP).copy()
    agg = (
        df.groupby(["id_student", "code_module", "code_presentation"])
        .apply(_final_feats)
        .reset_index()
    )

    p25   = agg["mean_score"].quantile(0.25)
    p50   = agg["mean_score"].quantile(0.50)
    p75   = agg["mean_score"].quantile(0.75)
    cp15  = agg["mean_clicks"].quantile(0.15)
    cp50  = agg["mean_clicks"].quantile(0.50)
    # Anxious threshold: must have notably high help-seeking (p85 of cohort)
    # Avoids over-labeling students who occasionally use forums
    hp85  = agg["mean_help"].quantile(0.85)

    agg["archetype_final"] = agg.apply(
        lambda r: _label_final(r, p25, p50, p75, cp15, cp50, hp85), axis=1
    )
    return agg


def _final_feats(grp):
    grp    = grp.sort_values("week")
    n      = len(grp)
    mid    = max(1, n // 2)
    scores = grp["engagement_score"].values
    clicks = grp["weekly_clicks"].values         if "weekly_clicks"       in grp.columns else np.zeros(n)
    help_  = grp["help_seeking_clicks"].values   if "help_seeking_clicks" in grp.columns else np.zeros(n)

    first_half  = scores[:mid].mean()
    second_half = scores[mid:].mean() if n > mid else first_half
    slope       = float(np.polyfit(np.arange(n), scores, 1)[0]) if n >= 2 else 0.0

    # Peak archetype counts from real-time labels
    arch_counts = grp["archetype_week"].value_counts() if "archetype_week" in grp.columns else pd.Series(dtype=int)

    return pd.Series({
        "mean_score":       scores.mean(),
        "first_half":       first_half,
        "second_half":      second_half,
        "recovery":         second_half - first_half,
        "full_slope":       slope,
        "score_std":        scores.std(),
        "mean_clicks":      clicks.mean(),
        "mean_help":        help_.mean(),
        "n_weeks":          n,
        "dominant_archetype": arch_counts.index[0] if len(arch_counts) > 0 else "Steady Engager",
    })


def _label_final(row, p25, p50, p75, cp15, cp50, hp85):
    s       = row["mean_score"]
    clicks  = row["mean_clicks"]
    rec     = row["recovery"]
    slope   = row["full_slope"]
    help_   = row["mean_help"]

    high = s >= p75
    mid  = p25 <= s < p75
    low  = s < p25

    # 1. Ghost — barely present throughout the course
    if clicks <= cp15:
        return "Ghost"

    # 2. Late Recoverer — clear first-half dip, genuine second-half comeback
    #    (Requires full-course view — not knowable at week 6)
    if row["first_half"] < p50 and row["second_half"] >= p50 and rec > 10:
        return "Late Recoverer"

    # 3. Burnt-out Achiever — strong start, significant second-half decline
    #    (Requires full-course view — not knowable at week 6)
    if row["first_half"] >= p50 and row["second_half"] < p50 and rec < -10:
        return "Burnt-out Achiever"

    # 4. Early Dropout — persistently low score with declining slope
    if low and slope < -0.4:
        return "Early Dropout"

    # 5. Struggling — low score but still present (not ghosting, not declining rapidly)
    if low:
        return "Struggling"

    # 6. Anxious — above median but top-15% help-seeking (cohort-relative)
    if (high or mid) and help_ >= hp85 and hp85 > 0:
        return "Anxious"

    # 7. Coasting — above median score with below-median clicks
    if (high or mid) and clicks < cp50:
        return "Coasting"

    # 8. Steady Engager — default for stable, above-median students
    return "Steady Engager"
