"""
features_v2.py — Task 1v2: Behavioral Feature Engineering

10 scored features across 4 purely behavioral domains.
Assessment SCORE LEVEL removed — only submission BEHAVIOR retained.
deep_content and active_clicks are mutually exclusive (no double-counting).

Domains
-------
VLE Presence  (35%): weekly_clicks, days_active, login_consistency
VLE Quality   (25%): activity_diversity, deep_content_ratio, active_clicks_ratio
Trajectory    (20%): trend_slope, engagement_momentum
Submission    (20%): completion_rate, submission_timeliness

Auxiliary (archetype classification only, NOT in score):
  help_seeking_clicks, comeback_signal
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress

DATA_DIR = ".."
GROUP    = ["id_student", "code_module", "code_presentation", "week"]

# Mutually exclusive activity buckets
DEEP_TYPES   = {"oucontent", "resource"}                                      # passive content study
ACTIVE_TYPES = {"quiz", "forumng", "oucollaborate", "ouelluminate", "questionnaire"}  # interactive
HELP_TYPES   = {"forumng", "oucollaborate", "ouelluminate", "questionnaire"}  # collaborative


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir=DATA_DIR):
    print("Loading CSVs (studentVle is ~10 M rows, ~30 s)...")
    vle      = pd.read_csv(f"{data_dir}/studentVle.csv")
    vle_meta = pd.read_csv(f"{data_dir}/vle.csv")
    assess   = pd.read_csv(f"{data_dir}/assessments.csv")
    st_ass   = pd.read_csv(f"{data_dir}/studentAssessment.csv")
    st_info  = pd.read_csv(f"{data_dir}/studentInfo.csv")
    st_reg   = pd.read_csv(f"{data_dir}/studentRegistration.csv")
    print("Done.")
    return vle, vle_meta, assess, st_ass, st_info, st_reg


def add_week(df, date_col="date"):
    """Convert relative-day column to integer week number (week 0 = days 0–6)."""
    df = df.copy()
    df["week"] = df[date_col] // 7
    return df


def _enrich_vle(vle, vle_meta):
    """Merge activity_type onto VLE clicks once. Pass the result to all quality features."""
    return vle.merge(vle_meta[["id_site", "activity_type"]], on="id_site", how="left")


def _non_exam(assess):
    return assess[assess["assessment_type"] != "Exam"].copy()


def _slope(arr):
    """Linear slope of a numpy array (used inside rolling.apply)."""
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr))
    return float(linregress(x, arr).slope)


# ---------------------------------------------------------------------------
# Domain 1 — VLE Presence
# ---------------------------------------------------------------------------

def feat_weekly_clicks(vle):
    """F1 — Total clicks per student per week. Primary presence signal."""
    df = add_week(vle)
    return (
        df.groupby(GROUP)["sum_click"]
        .sum().reset_index()
        .rename(columns={"sum_click": "weekly_clicks"})
    )


def feat_days_active(vle):
    """F2 — Distinct days with ≥1 click per week. Distributed practice vs cramming."""
    df = add_week(vle)
    return (
        df.groupby(GROUP)["date"]
        .nunique().reset_index()
        .rename(columns={"date": "days_active"})
    )


def feat_login_consistency(weekly_clicks_df, window=3):
    """
    F3 — Stability of weekly click volume over a rolling window.
    CV = std / mean over [W-2, W-1, W]. Consistency = 1 - CV (clipped to [0,1]).
    First two weeks: neutral (0.5) because insufficient history.
    """
    df = weekly_clicks_df.sort_values(GROUP).copy()

    def _c(x):
        roll_mean = x.rolling(window, min_periods=2).mean()
        roll_std  = x.rolling(window, min_periods=2).std().fillna(0)
        # When mean is zero the student was inactive — consistency = 0 (not neutral 0.5)
        cv        = roll_std / (roll_mean + 1e-9)
        cv        = cv.where(roll_mean > 0, 0.0)
        con       = (1 - cv.clip(0, 1)).where(roll_mean > 0, 0.0)
        # first observation: no history → neutral
        if len(con) > 0:
            con.iloc[0] = 0.5
        return con

    df["login_consistency"] = (
        df.groupby(["id_student", "code_module", "code_presentation"])["weekly_clicks"]
        .transform(_c)
    )
    return df[GROUP + ["login_consistency"]]


# ---------------------------------------------------------------------------
# Domain 2 — VLE Quality  (DEEP_TYPES ∩ ACTIVE_TYPES = ∅)
# ---------------------------------------------------------------------------

def feat_activity_diversity(vle_enriched):
    """F4 — Distinct activity_type count per week. Breadth of engagement."""
    df = add_week(vle_enriched)
    return (
        df.groupby(GROUP)["activity_type"]
        .nunique().reset_index()
        .rename(columns={"activity_type": "activity_diversity"})
    )


def feat_deep_content_ratio(vle_enriched):
    """
    F5 — Clicks on deep-study types (oucontent, resource) / total clicks.
    DEEP_TYPES and ACTIVE_TYPES are disjoint — no double-counting.
    """
    df = add_week(vle_enriched)
    df["deep"] = df["activity_type"].isin(DEEP_TYPES).astype(int) * df["sum_click"]
    grp = df.groupby(GROUP).agg(total=("sum_click", "sum"), deep=("deep", "sum")).reset_index()
    grp["deep_content_ratio"] = grp["deep"] / (grp["total"] + 1e-9)
    return grp[GROUP + ["deep_content_ratio"]]


def feat_active_clicks_ratio(vle_enriched):
    """
    F6 — Clicks on interactive types (quiz, forum, collaborate) / total clicks.
    ACTIVE_TYPES does NOT include oucontent/resource — no overlap with F5.
    """
    df = add_week(vle_enriched)
    df["active"] = df["activity_type"].isin(ACTIVE_TYPES).astype(int) * df["sum_click"]
    grp = df.groupby(GROUP).agg(total=("sum_click", "sum"), active=("active", "sum")).reset_index()
    grp["active_clicks_ratio"] = grp["active"] / (grp["total"] + 1e-9)
    return grp[GROUP + ["active_clicks_ratio"]]


# ---------------------------------------------------------------------------
# Domain 3 — Trajectory
# ---------------------------------------------------------------------------

def feat_trend_slope(weekly_clicks_df, window=3):
    """
    F7 — OLS slope of weekly_clicks over a rolling window=3.
    Fully vectorized: for x=[0,1,2], slope = (y2 - y0) / 2.
    Avoids slow rolling.apply on 600K+ rows.
    """
    df = weekly_clicks_df.sort_values(GROUP).copy()
    gk = ["id_student", "code_module", "code_presentation"]
    g  = df.groupby(gk)["weekly_clicks"]

    cur  = df["weekly_clicks"]
    lag1 = g.shift(1)
    lag2 = g.shift(2)

    has2 = lag2.notna()
    has1 = lag1.notna() & ~has2

    slope = pd.Series(np.nan, index=df.index)
    slope.loc[has2] = (cur[has2].values - lag2[has2].values) / 2.0
    slope.loc[has1] = cur[has1].values - lag1[has1].values

    df["trend_slope"] = slope
    return df[GROUP + ["trend_slope"]]


def feat_engagement_momentum(weekly_clicks_df):
    """
    F8 — Week-over-week % change in clicks, clipped to ±200%. Captures acceleration.
    When previous week = 0, pct_change() returns inf which clips to +2 (max signal).
    Fix: set momentum to 0 when prior week had zero clicks — no signal, not max signal.
    """
    df = weekly_clicks_df.sort_values(GROUP).copy()
    gk = ["id_student", "code_module", "code_presentation"]

    prev_clicks = df.groupby(gk)["weekly_clicks"].shift(1)
    raw_momentum = df["weekly_clicks"].sub(prev_clicks).div(prev_clicks.abs() + 1e-9).clip(-2, 2)
    # Zero out momentum when prior week had no activity (undefined, not maximum)
    raw_momentum = raw_momentum.where(prev_clicks.fillna(0) > 0, 0.0)

    df["engagement_momentum"] = raw_momentum.fillna(0.0)
    return df[GROUP + ["engagement_momentum"]]


# ---------------------------------------------------------------------------
# Domain 4 — Submission Behavior
# ---------------------------------------------------------------------------

def feat_completion_rate(st_ass, assess, st_info):
    """
    F9 — Cumulative fraction of due TMA/CMAs submitted by each student by week W.
    Non-submitters get 0.0 (not cohort median) — this is intentional.
    Rows only produced for assessment-due-weeks; scoring.py forward-fills the rest.
    """
    non_exam = _non_exam(assess).copy()
    non_exam = add_week(non_exam, "date").rename(columns={"week": "due_week"})

    submitted = st_ass.merge(
        non_exam[["id_assessment", "code_module", "code_presentation", "due_week"]],
        on="id_assessment", how="inner"
    )

    all_students = st_info[["id_student", "code_module", "code_presentation"]].drop_duplicates()
    records = []

    for (mod, pres), grp_assess in non_exam.groupby(["code_module", "code_presentation"]):
        cohort = all_students.loc[
            (all_students["code_module"] == mod) & (all_students["code_presentation"] == pres),
            "id_student"
        ].values

        sub_here = submitted.loc[
            (submitted["code_module"] == mod) & (submitted["code_presentation"] == pres)
        ]

        for week in sorted(grp_assess["due_week"].unique()):
            due_so_far = int((grp_assess["due_week"] <= week).sum())
            if due_so_far == 0:
                continue

            sub_counts = (
                sub_here[sub_here["due_week"] <= week]
                .groupby("id_student").size()
                .reindex(cohort, fill_value=0)
            )

            records.append(pd.DataFrame({
                "id_student":        cohort,
                "code_module":       mod,
                "code_presentation": pres,
                "week":              week,
                "completion_rate":   (sub_counts.values / due_so_far).clip(0, 1),
            }))

    if not records:
        return pd.DataFrame(columns=GROUP + ["completion_rate"])
    return pd.concat(records, ignore_index=True)


def feat_submission_timeliness(st_ass, assess):
    """
    F10 — Mean days late per week for submitted TMA/CMAs (due_date - submit_date).
    Negative = submitted early (good). Missing = no submission that week → NaN → forward-filled.
    INVERTED in scoring (earlier = better).
    """
    non_exam = _non_exam(assess)
    df = st_ass.merge(
        non_exam[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment", how="inner"
    )
    df["days_late"] = df["date_submitted"] - df["date"]
    df = add_week(df, "date")
    return (
        df.groupby(GROUP)["days_late"]
        .mean().reset_index()
        .rename(columns={"days_late": "submission_timeliness"})
    )


# ---------------------------------------------------------------------------
# Auxiliary — archetype use only (not in score)
# ---------------------------------------------------------------------------

def feat_help_seeking(vle_enriched):
    """AUX-1 — Clicks on collaborative/help types. Used for Anxious archetype detection."""
    df = add_week(vle_enriched)
    df["help"] = df["activity_type"].isin(HELP_TYPES).astype(int) * df["sum_click"]
    return (
        df.groupby(GROUP)["help"]
        .sum().reset_index()
        .rename(columns={"help": "help_seeking_clicks"})
    )


def feat_comeback_signal(weekly_clicks_df):
    """
    AUX-2 — 1 if clicks recovered after a below-personal-median week.
    Fully vectorized: no Python loop over rows.
    """
    df = weekly_clicks_df.sort_values(GROUP).copy()
    gk = ["id_student", "code_module", "code_presentation"]

    median_clicks = df.groupby(gk)["weekly_clicks"].transform("median")
    below         = (df["weekly_clicks"] < median_clicks).astype(int)
    prev_below    = df.groupby(gk)["weekly_clicks"].transform(lambda x: (x < x.median()).astype(int)).shift(1).fillna(0)
    prev_clicks   = df.groupby(gk)["weekly_clicks"].shift(1).fillna(0)

    df["comeback_signal"] = (
        (prev_below == 1) & (df["weekly_clicks"] > prev_clicks)
    ).astype(int)

    return df[GROUP + ["comeback_signal"]]


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_feature_table(vle, vle_meta, assess, st_ass, st_info):
    """
    Compute all 12 features and merge into one wide table keyed on GROUP.
    _enrich_vle called once to avoid repeated 10M-row merges.
    """
    print("Enriching VLE with activity types (single merge)...")
    vle_e = _enrich_vle(vle, vle_meta)

    print("Domain 1 — VLE Presence...")
    wk_clicks   = feat_weekly_clicks(vle)
    days_act    = feat_days_active(vle)
    consistency = feat_login_consistency(wk_clicks)

    print("Domain 2 — VLE Quality...")
    diversity   = feat_activity_diversity(vle_e)
    deep_cont   = feat_deep_content_ratio(vle_e)
    active_rat  = feat_active_clicks_ratio(vle_e)

    print("Domain 3 — Trajectory...")
    slope       = feat_trend_slope(wk_clicks)
    momentum    = feat_engagement_momentum(wk_clicks)

    print("Domain 4 — Submission Behavior...")
    compl_rate  = feat_completion_rate(st_ass, assess, st_info)
    timeliness  = feat_submission_timeliness(st_ass, assess)

    print("Auxiliary features...")
    help_seek   = feat_help_seeking(vle_e)
    comeback    = feat_comeback_signal(wk_clicks)

    print("Merging all features...")
    df = (
        wk_clicks
        .merge(days_act,    on=GROUP, how="left")
        .merge(consistency, on=GROUP, how="left")
        .merge(diversity,   on=GROUP, how="left")
        .merge(deep_cont,   on=GROUP, how="left")
        .merge(active_rat,  on=GROUP, how="left")
        .merge(slope,       on=GROUP, how="left")
        .merge(momentum,    on=GROUP, how="left")
        .merge(compl_rate,  on=GROUP, how="left")
        .merge(timeliness,  on=GROUP, how="left")
        .merge(help_seek,   on=GROUP, how="left")
        .merge(comeback,    on=GROUP, how="left")
    )

    # Forward-fill submission features within each student.
    # Assessments only exist in certain weeks; carrying last known value forward
    # is more honest than treating non-assessment weeks as missing.
    df = df.sort_values(GROUP)
    for col in ["completion_rate", "submission_timeliness"]:
        df[col] = (
            df.groupby(["id_student", "code_module", "code_presentation"])[col]
            .transform(lambda x: x.ffill())
        )

    print(f"Feature table: {len(df):,} rows × {len(df.columns)} columns")
    return df
