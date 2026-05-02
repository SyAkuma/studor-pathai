"""
cf_vle.py — VLE activity-profile collaborative filtering.

Key insight: 87.7% of students took only 1 module, so user-user CF on module
history is almost useless. But every student has a within-module behavioral
fingerprint — how they distribute their clicks across activity types. This
profile is comparable across modules after within-module normalisation and works
even for students with a single module history.

Similarity metric: Pearson correlation (implemented as mean-centering + cosine).
Why Pearson over cosine:
  A heavy user (10,000 clicks) and a light user (300 clicks) with the same
  activity-type distribution are behaviorally similar — same style, different
  intensity. Cosine would consider them similar only if magnitudes also match.
  Pearson subtracts the row mean first, so only the relative distribution matters.

Why within-module normalisation:
  oucontent constitutes different fractions of total VLE sites across modules.
  Subtracting the module mean fraction removes the effect of module structure —
  what remains is each student's preference relative to their peers in the same
  module and presentation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

TOP_N_NEIGHBOURS = 20


def build_vle_profiles(vle, vle_meta, si):
    """
    Returns:
      profiles  : DataFrame, index=id_student, columns=activity_type fractions
                  (within-module normalised, log-click weighted average across modules)
      weights   : Series, index=id_student — total clicks (for weighting quality)
    """
    print("Merging VLE with activity types...")
    merged = vle.merge(
        vle_meta[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"], how="inner"
    )

    print("Computing per-student per-module activity-type profiles...")
    # Total clicks per student per activity type per module-presentation
    agg = (
        merged.groupby(
            ["id_student", "code_module", "code_presentation", "activity_type"]
        )["sum_click"].sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    mod_group = ["code_module", "code_presentation"]
    act_cols = [c for c in agg.columns if c not in ["id_student"] + mod_group]

    # Total clicks per student per module (for weighting)
    agg["_total_clicks"] = agg[act_cols].sum(axis=1)

    # Fraction of clicks on each activity type (per student per module)
    agg[act_cols] = agg[act_cols].div(agg["_total_clicks"].replace(0, 1), axis=0)

    # Within-module normalisation: subtract module mean fraction
    # After this, a positive value means the student used that activity type MORE
    # than the average student in that module
    for ac in act_cols:
        mod_means = agg.groupby(mod_group)[ac].transform("mean")
        agg[f"norm_{ac}"] = agg[ac] - mod_means

    norm_cols = [f"norm_{c}" for c in act_cols]

    # Aggregate across modules per student, weighted by log(1 + total_clicks)
    # Students with more total clicks get more weight — their profile is more stable
    agg["_log_weight"] = np.log1p(agg["_total_clicks"])

    profiles_list = []
    weights_list  = []
    for sid, grp in agg.groupby("id_student"):
        w = grp["_log_weight"].values
        w_sum = w.sum()
        if w_sum == 0:
            weighted = grp[norm_cols].mean()
        else:
            weighted = (grp[norm_cols].values * w[:, None]).sum(axis=0) / w_sum
        profiles_list.append({"id_student": sid, **dict(zip(norm_cols, weighted))})
        weights_list.append({"id_student": sid, "total_clicks": grp["_total_clicks"].sum()})

    profiles = pd.DataFrame(profiles_list).set_index("id_student")
    weights  = pd.DataFrame(weights_list).set_index("id_student")["total_clicks"]

    print(f"VLE profiles built: {len(profiles):,} students × {len(profiles.columns)} activity dims")
    return profiles, weights, act_cols


def compute_neighbours(profiles, weights, top_n=TOP_N_NEIGHBOURS):
    """
    Find top_n nearest neighbours per student using Pearson correlation distance.
    Pearson = mean-centering rows + cosine similarity.

    Implementation: subtract row mean (centering), normalise rows to unit length,
    use NearestNeighbors with cosine metric (equivalent to Pearson after centering).

    Returns:
      neighbours : DataFrame, index=id_student, columns=['nbr_0'...'nbr_N', 'sim_0'...'sim_N']
    """
    print("Computing nearest neighbours (Pearson profile similarity)...")
    X = profiles.values.copy()

    # Mean-center rows (Pearson step)
    X = X - X.mean(axis=1, keepdims=True)

    # Replace zero-variance rows with zeros (students who only used 1 activity type)
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.where(row_norms > 0, X / row_norms, 0)

    # NearestNeighbors: cosine on mean-centred unit vectors = Pearson correlation
    nn = NearestNeighbors(n_neighbors=top_n + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # distances are cosine distances (1 - similarity), convert to similarity
    similarities = 1 - distances

    student_ids = profiles.index.tolist()
    records = []
    for i, sid in enumerate(student_ids):
        row = {"id_student": sid}
        for rank, (j, sim) in enumerate(zip(indices[i][1:], similarities[i][1:])):
            row[f"nbr_{rank}"] = student_ids[j]
            row[f"sim_{rank}"] = round(float(sim), 4)
        records.append(row)

    neighbours = pd.DataFrame(records).set_index("id_student")
    print(f"Neighbours computed: {len(neighbours):,} students, top {top_n} each")
    return neighbours


def build_si_index(si):
    """
    Pre-index studentInfo by id_student for O(1) lookup in recommend_cf.
    Call once before batch recommendations.
    Returns dict: {id_student: [(code_module, final_result), ...]}
    """
    outcome_weight = {"Distinction": 1.2, "Pass": 1.0}
    idx = {}
    for sid, grp in si.groupby("id_student"):
        rows = [(r["code_module"], r["final_result"])
                for _, r in grp.iterrows()
                if r["final_result"] in outcome_weight]
        if rows:
            idx[sid] = rows
    return idx


def recommend_cf(id_student, already_taken, neighbours, si, weights, top_k=3, si_index=None):
    """
    Recommend modules using VLE-profile collaborative filtering.

    si_index: optional pre-built index from build_si_index() — pass this for
    batch calls to avoid O(N) si scan per neighbour per student.
    """
    if id_student not in neighbours.index:
        return []

    nbr_row = neighbours.loc[id_student]
    n_cols  = len([c for c in nbr_row.index if c.startswith("nbr_")])

    outcome_weight = {"Distinction": 1.2, "Pass": 1.0}
    votes = {}

    for rank in range(n_cols):
        nbr_id = nbr_row.get(f"nbr_{rank}")
        sim    = nbr_row.get(f"sim_{rank}", 0)
        if pd.isna(nbr_id) or sim <= 0:
            continue

        click_weight = np.log1p(weights.get(nbr_id, 1))

        # Use pre-built index if available, else fall back to si scan
        if si_index is not None:
            nbr_mods = si_index.get(int(nbr_id), [])
        else:
            nbr_df   = si[si["id_student"] == nbr_id][["code_module", "final_result"]]
            nbr_mods = [(r["code_module"], r["final_result"]) for _, r in nbr_df.iterrows()]

        for mod, result in nbr_mods:
            if result not in outcome_weight or mod in already_taken:
                continue
            ow = outcome_weight[result]
            votes[mod] = votes.get(mod, 0) + sim * ow * click_weight

    if not votes:
        return []

    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:top_k]
    max_vote = sorted_votes[0][1]
    return [
        {"module": m, "score": round(v / max_vote, 4), "method": "VLE-CF"}
        for m, v in sorted_votes
    ]
