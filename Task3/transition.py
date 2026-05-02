"""
transition.py — Module transition Markov chain recommender.

Builds P(next_module | current_module, last_outcome).
Uses the 3,538 students who took 2+ modules.

Why condition on outcome:
  A student who passed CCC then took EEE is making a progression decision.
  A student who withdrew from CCC then took BBB may be looking for something different.
  These are structurally different transitions — mixing them biases toward raw popularity.

Why NOT condition on IMD:
  Only 3,808 transitions across 7 modules × 4 outcomes already gives median ~79 obs per cell.
  Adding IMD halves that to ~30. The signal is too thin to justify the complexity.

Laplace smoothing (pseudocount=1):
  Some (current, outcome) combinations have few observations.
  Smoothing prevents zero-probability next-modules and allows graceful fallback.

Fallback: if (current, outcome) has fewer than 5 raw observations,
  fall back to P(next | current) ignoring outcome entirely.
"""

import pandas as pd

PRES_ORDER = {"2013B": 0, "2013J": 1, "2014B": 2, "2014J": 3}
SMOOTHING  = 1


def build_transition_matrix(si):
    """
    Build transition probability table from students with 2+ modules.

    Returns:
      trans_df : DataFrame — P(next | current, outcome) with Laplace smoothing
      marg_df  : DataFrame — P(next | current) fallback, ignoring outcome
    """
    df = si.copy()
    df["pres_ord"] = df["code_presentation"].map(PRES_ORDER).fillna(0)
    df = df.sort_values(["id_student", "pres_ord"])

    all_modules = sorted(si["code_module"].unique())

    # Build consecutive (current → next) pairs
    records = []
    for sid, grp in df.groupby("id_student"):
        mods     = grp["code_module"].tolist()
        outcomes = grp["final_result"].tolist()
        for i in range(len(mods) - 1):
            records.append({
                "id_student":     sid,
                "current_module": mods[i],
                "last_outcome":   outcomes[i],
                "next_module":    mods[i + 1],
            })

    pairs = pd.DataFrame(records)
    n_students = pairs["id_student"].nunique()
    n_pairs    = len(pairs)
    print(f"Transition matrix built: {n_students:,} students, {n_pairs:,} transitions")
    top = (
        pairs.groupby(["current_module", "next_module"])
        .size().reset_index(name="n")
        .sort_values("n", ascending=False).head(5)
    )
    print("Top 5 transitions:")
    print(top.to_string(index=False))

    # ── Conditioned: P(next | current, outcome) ──────────────────────────────
    counts = (
        pairs.groupby(["current_module", "last_outcome", "next_module"])
        .size().reset_index(name="count")
    )
    combos = counts[["current_module", "last_outcome"]].drop_duplicates()
    rows = []
    for _, row in combos.iterrows():
        for nm in all_modules:
            ex = counts[
                (counts["current_module"] == row["current_module"]) &
                (counts["last_outcome"]   == row["last_outcome"]) &
                (counts["next_module"]    == nm)
            ]
            c = int(ex["count"].sum()) if len(ex) > 0 else 0
            rows.append({
                "current_module": row["current_module"],
                "last_outcome":   row["last_outcome"],
                "next_module":    nm,
                "count":          c + SMOOTHING,
            })
    trans_df = pd.DataFrame(rows)
    totals = trans_df.groupby(["current_module", "last_outcome"])["count"].transform("sum")
    trans_df["prob"] = trans_df["count"] / totals

    # ── Marginal fallback: P(next | current) ─────────────────────────────────
    marg_counts = (
        pairs.groupby(["current_module", "next_module"])
        .size().reset_index(name="count")
    )
    marg_combos = marg_counts["current_module"].unique()
    marg_rows = []
    for cur in marg_combos:
        for nm in all_modules:
            ex = marg_counts[
                (marg_counts["current_module"] == cur) &
                (marg_counts["next_module"]    == nm)
            ]
            c = int(ex["count"].sum()) if len(ex) > 0 else 0
            marg_rows.append({
                "current_module": cur,
                "next_module":    nm,
                "count":          c + SMOOTHING,
            })
    marg_df = pd.DataFrame(marg_rows)
    marg_totals = marg_df.groupby("current_module")["count"].transform("sum")
    marg_df["prob"] = marg_df["count"] / marg_totals

    return trans_df, marg_df


def recommend_markov(current_module, last_outcome, already_taken, trans_df, marg_df, top_k=3):
    """
    Recommend top_k next modules given current module and last outcome.

    Falls back to marginal P(next | current) if the conditioned combination
    has fewer than 5 raw observations.
    """
    cond = trans_df[
        (trans_df["current_module"] == current_module) &
        (trans_df["last_outcome"]   == last_outcome)
    ].copy()

    # Fall back to marginal if conditioned signal is thin
    if len(cond) == 0 or cond["count"].sum() <= (5 + len(cond) * SMOOTHING):
        cond = marg_df[marg_df["current_module"] == current_module].copy()

    if len(cond) == 0:
        return []

    # Allow retake only if student failed or withdrew
    allow_same = last_outcome in ("Fail", "Withdrawn")
    if allow_same:
        cond = cond[~cond["next_module"].isin(
            [m for m in already_taken if m != current_module]
        )]
    else:
        cond = cond[~cond["next_module"].isin(already_taken)]

    recs = (
        cond.sort_values("prob", ascending=False)
        .head(top_k)[["next_module", "prob"]]
        .rename(columns={"next_module": "module", "prob": "score"})
    )
    recs["method"] = "Markov"
    return recs.to_dict("records")
