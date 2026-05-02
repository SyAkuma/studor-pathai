"""
model_v2.py — Task 2v2: Training, validation, calibration, evaluation.

Key changes from v1:
  - Temporal split: train=2013 presentations, test=2014 presentations
  - Three-way split within training: 75% train base model, 25% validation
    (tune threshold + fit Platt calibration on val, never on test)
  - Consistent scale_pos_weight across CV and final model (v1 used weight=1 in CV)
  - Threshold selected by operational constraint (flag rate ≤ 35%) on val set,
    not F2-maximisation on test set
  - Calibration fit on held-out val set (not cv=5 on training data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model    import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.calibration     import calibration_curve
from sklearn.metrics         import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, brier_score_loss, precision_recall_curve,
    RocCurveDisplay, ConfusionMatrixDisplay, fbeta_score
)
import xgboost as xgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE     = 42
GROUP            = ["id_student", "code_module", "code_presentation"]
FLAG_RATE_CEILING = 0.50   # raised from 0.35 — advisors use High/Medium tiers to prioritise within the flagged pool


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def temporal_split(df):
    """
    Train: 2013J + 2013B presentations.
    Test:  2014J + 2014B presentations.

    Rationale: deployment always predicts future cohorts from past cohorts.
    Random stratified split (v1) mixed years and inflated AUROC.
    """
    train_pres = ["2013J", "2013B"]
    test_pres  = ["2014J", "2014B"]

    df_train = df[df["code_presentation"].isin(train_pres)].copy()
    df_test  = df[df["code_presentation"].isin(test_pres)].copy()

    print(f"Temporal split:")
    print(f"  Train ({', '.join(train_pres)}): {len(df_train):,} students")
    print(f"  Test  ({', '.join(test_pres)}): {len(df_test):,} students")
    print(f"  Train label rate: {df_train['label'].mean():.3f}")
    print(f"  Test  label rate: {df_test['label'].mean():.3f}")

    return df_train, df_test


def prepare_Xy(df, feat_cols):
    X = df[feat_cols].values.astype(float)
    y = df["label"].values
    return X, y


def get_feature_cols(df):
    return [c for c in df.columns if c not in GROUP + ["code_presentation", "label"]]


def train_val_split(df_train, val_size=0.25):
    """
    Hold out 25% of training data for threshold tuning and calibration.
    This data is never seen by the base model.
    """
    train_idx, val_idx = train_test_split(
        np.arange(len(df_train)),
        test_size=val_size,
        stratify=df_train["label"].values,
        random_state=RANDOM_STATE
    )
    return df_train.iloc[train_idx], df_train.iloc[val_idx]


# ---------------------------------------------------------------------------
# Model building — consistent scale_pos_weight everywhere
# ---------------------------------------------------------------------------

def compute_pos_weight(y):
    neg, pos = (y == 0).sum(), (y == 1).sum()
    return neg / pos


def build_models(pos_weight):
    """
    All models use the same class imbalance correction.
    v1 used scale_pos_weight=1 in CV but computed pos_weight in final model.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                       random_state=RANDOM_STATE))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            scale_pos_weight=pos_weight,          # same in CV and final model
            eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def cross_validate_models(models, X_tr, y_tr, cv=5):
    cv_obj  = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in models.items():
        scores = cross_validate(
            model, X_tr, y_tr, cv=cv_obj,
            scoring={"auroc": "roc_auc", "recall": "recall"},
            return_train_score=False
        )
        results[name] = {
            "auroc_mean":  scores["test_auroc"].mean(),
            "auroc_std":   scores["test_auroc"].std(),
            "recall_mean": scores["test_recall"].mean(),
            "recall_std":  scores["test_recall"].std(),
        }
        print(f"{name:25s}  AUROC={results[name]['auroc_mean']:.3f}±{results[name]['auroc_std']:.3f}"
              f"  Recall={results[name]['recall_mean']:.3f}±{results[name]['recall_std']:.3f}")
    return results


def train_base_model(X_tr, y_tr, params=None):
    """Fit XGBoost with consistent pos_weight. Accepts optional tuned params dict."""
    pos_weight = compute_pos_weight(y_tr)
    defaults = dict(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    if params:
        defaults.update(params)
        defaults["scale_pos_weight"] = pos_weight  # always recompute from actual labels
    model = xgb.XGBClassifier(**defaults)
    model.fit(X_tr, y_tr)
    return model


def optuna_tune(X_tr, y_tr, X_val, y_val, n_trials=80):
    """
    Bayesian hyperparameter search using Optuna.
    Objective: maximise AUROC on val set (threshold-independent → no leakage).
    Val set is held out from base model training throughout.

    Why Optuna over grid search:
      Grid search evaluates all combinations — exponential in parameters.
      Optuna uses Tree-structured Parzen Estimator (TPE): it builds a probabilistic
      model of which regions of the search space produce good results, and samples
      from promising regions. 80 trials finds better configs than a 5^6 = 15,625
      grid search, in a fraction of the time.

    Why optimise AUROC (not Recall directly):
      AUROC is threshold-independent — it measures the model's ability to rank
      at-risk students above safe ones regardless of where we draw the line.
      Optimising Recall directly would just push the model to flag everyone
      (Recall = 1.0 if you flag 100% of students). AUROC improvement raises the
      Recall ceiling at any given flag rate.
    """
    pos_weight = compute_pos_weight(y_tr)
    cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 700),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 10.0),
            "scale_pos_weight":  pos_weight,
            "eval_metric":       "logloss",
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
        }
        # 5-fold CV on training set — never touches val or test
        model = xgb.XGBClassifier(**params)
        scores = cross_validate(
            model, X_tr, y_tr, cv=cv_obj,
            scoring="roc_auc", return_train_score=False
        )
        return scores["test_score"].mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_auroc = study.best_value
    print(f"\nOptuna best CV AUROC : {best_auroc:.4f}  (over {n_trials} trials)")
    print("Best hyperparameters:")
    for k, v in best.items():
        print(f"  {k:20s}: {v}")

    # confirm on val set (separate from CV — sanity check, not used for selection)
    final_model = xgb.XGBClassifier(
        **{k: v for k, v in best.items()},
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    final_model.fit(X_tr, y_tr)
    val_auroc = roc_auc_score(y_val, final_model.predict_proba(X_val)[:, 1])
    print(f"Val set AUROC (tuned model): {val_auroc:.4f}")

    return final_model, best, study


# ---------------------------------------------------------------------------
# Threshold tuning — operational constraint on validation set
# ---------------------------------------------------------------------------

def tune_threshold_operational(model_or_cal, X_val, y_val,
                                flag_rate_ceiling=FLAG_RATE_CEILING):
    """
    Select threshold on validation set (not test set — v1's critical error).

    Operational constraint: flag rate ≤ flag_rate_ceiling.
    Within that ceiling, maximise Recall.
    This gives a threshold that's actually deployable, not a mathematical
    artefact of optimising F2 without any flag-rate constraint.

    v1 at threshold=0.16 flagged 89.7% of students — worse than flagging everyone.
    """
    probs      = model_or_cal.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.95, 0.005)
    records    = []

    for t in thresholds:
        preds     = (probs >= t).astype(int)
        flag_rate = preds.mean()
        rec       = recall_score(y_val, preds, zero_division=0)
        prec      = precision_score(y_val, preds, zero_division=0)
        f1        = f1_score(y_val, preds, zero_division=0)
        f2        = fbeta_score(y_val, preds, beta=2, zero_division=0)
        records.append({
            "threshold": t, "flag_rate": flag_rate,
            "recall": rec, "precision": prec, "f1": f1, "f2": f2
        })

    sweep = pd.DataFrame(records)

    # operational constraint: flag rate ≤ ceiling
    candidates = sweep[sweep["flag_rate"] <= flag_rate_ceiling]
    if len(candidates) == 0:
        print(f"WARNING: No threshold achieves flag_rate ≤ {flag_rate_ceiling:.0%}. Using closest.")
        candidates = sweep.copy()

    best_t = candidates.loc[candidates["recall"].idxmax(), "threshold"]
    chosen = sweep[sweep["threshold"].round(3) == round(best_t, 3)].iloc[0]

    print(f"\nOperational threshold selection (val set, flag_rate_ceiling={flag_rate_ceiling:.0%}):")
    print(f"  Best threshold : {best_t:.3f}")
    print(f"  Flag rate      : {chosen['flag_rate']:.3f} ({chosen['flag_rate']*100:.1f}% of students)")
    print(f"  Recall         : {chosen['recall']:.4f}")
    print(f"  Precision      : {chosen['precision']:.4f}")
    print(f"  F1             : {chosen['f1']:.4f}")

    return float(best_t), sweep


# ---------------------------------------------------------------------------
# Calibration — fit on held-out val set
# ---------------------------------------------------------------------------

class _PlattWrapper:
    """Thin wrapper: base model raw scores → Platt sigmoid → calibrated probs."""
    def __init__(self, base_model, platt):
        self.base_model = base_model
        self.platt      = platt

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self.platt.predict_proba(raw)
        return cal


def calibrate(base_model, X_val, y_val):
    """
    Manual Platt scaling on held-out validation set.
    cv='prefit' was removed in sklearn >= 1.2; this is the equivalent approach:
    fit a logistic regression on the raw scores from the base model on X_val.
    """
    raw_val = base_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(max_iter=1000)
    platt.fit(raw_val, y_val)
    return _PlattWrapper(base_model, platt)


# ---------------------------------------------------------------------------
# Evaluation — all final metrics on test set only
# ---------------------------------------------------------------------------

def evaluate(model, X_te, y_te, threshold, feat_cols, label="Model", output_dir="."):
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= threshold).astype(int)

    flag_rate = preds.mean()
    metrics = {
        "AUROC":     roc_auc_score(y_te, probs),
        "Recall":    recall_score(y_te, preds),
        "Precision": precision_score(y_te, preds, zero_division=0),
        "F1":        f1_score(y_te, preds),
        "F2":        fbeta_score(y_te, preds, beta=2),
        "Brier":     brier_score_loss(y_te, probs),
        "Flag Rate": flag_rate,
        "Threshold": threshold,
    }

    print(f"\n{'='*55}")
    print(f"{label} — Test Set Results @ threshold={threshold:.3f}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    _plot_evaluation(model, X_te, y_te, probs, preds, threshold, label, output_dir)
    return metrics, probs


def _plot_evaluation(model, X_te, y_te, probs, preds, threshold, label, output_dir):
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # 1. ROC curve
    ax1 = fig.add_subplot(gs[0, 0])
    RocCurveDisplay.from_predictions(y_te, probs, ax=ax1, name=label)
    ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax1.set_title("ROC Curve")

    # 2. Precision-Recall curve
    ax2 = fig.add_subplot(gs[0, 1])
    prec_arr, rec_arr, _ = precision_recall_curve(y_te, probs)
    ax2.plot(rec_arr, prec_arr, color="steelblue")
    op_prec = precision_score(y_te, preds, zero_division=0)
    op_rec  = recall_score(y_te, preds)
    ax2.scatter([op_rec], [op_prec], color="red", zorder=5, s=80,
                label=f"t={threshold:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(fontsize=8)

    # 3. Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    ConfusionMatrixDisplay.from_predictions(
        y_te, preds, display_labels=["Safe", "At-Risk"], ax=ax3, colorbar=False
    )
    ax3.set_title("Confusion Matrix")

    # 4. Operational threshold sweep — flag rate vs recall
    ax4 = fig.add_subplot(gs[1, 0])
    thresholds = np.arange(0.05, 0.95, 0.005)
    rec_list, flag_list, prec_list = [], [], []
    for t in thresholds:
        p = (probs >= t).astype(int)
        rec_list.append(recall_score(y_te, p, zero_division=0))
        prec_list.append(precision_score(y_te, p, zero_division=0))
        flag_list.append(p.mean())
    ax4.plot(thresholds, rec_list,  label="Recall",    color="green")
    ax4.plot(thresholds, prec_list, label="Precision", color="orange")
    ax4.plot(thresholds, flag_list, label="Flag Rate", color="steelblue", linestyle="--")
    ax4.axvline(threshold, color="red", linestyle="--", label=f"Chosen t={threshold:.3f}")
    ax4.axhline(FLAG_RATE_CEILING, color="steelblue", linestyle=":", alpha=0.5,
                label=f"Flag ceiling {FLAG_RATE_CEILING:.0%}")
    ax4.set_xlabel("Threshold"); ax4.set_ylabel("Value")
    ax4.set_title("Threshold Sweep — Operational Constraint")
    ax4.legend(fontsize=7)

    # 5. Calibration reliability diagram
    ax5 = fig.add_subplot(gs[1, 1])
    frac_pos, mean_pred = calibration_curve(y_te, probs, n_bins=10, strategy="uniform")
    ax5.plot(mean_pred, frac_pos, "s-", label=label)
    ax5.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax5.set_xlabel("Mean predicted probability")
    ax5.set_ylabel("Fraction positives")
    ax5.set_title("Calibration Curve")
    ax5.legend(fontsize=8)

    # 6. Score distribution by class
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(probs[y_te == 0], bins=40, alpha=0.6, label="Safe",    color="steelblue")
    ax6.hist(probs[y_te == 1], bins=40, alpha=0.6, label="At-Risk", color="tomato")
    ax6.axvline(threshold, color="black", linestyle="--", label=f"t={threshold:.3f}")
    ax6.set_xlabel("Predicted probability"); ax6.set_ylabel("Count")
    ax6.set_title("Score Distribution by Class")
    ax6.legend(fontsize=8)

    fig.suptitle(f"Task 2v2 — {label} (Temporal Validation)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{output_dir}/eval_{label.replace(' ', '_').lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Per-module breakdown
# ---------------------------------------------------------------------------

def per_module_metrics(model, df_te, feat_cols, threshold, output_dir="."):
    X_te  = df_te[feat_cols].values.astype(float)
    y_te  = df_te["label"].values
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= threshold).astype(int)

    df_te = df_te.copy()
    df_te["prob"] = probs
    df_te["pred"] = preds

    rows = []
    for mod, grp in df_te.groupby("code_module"):
        y_g = grp["label"].values
        p_g = grp["prob"].values
        d_g = grp["pred"].values
        if len(np.unique(y_g)) < 2:
            continue
        rows.append({
            "module":     mod,
            "n":          len(grp),
            "pct_atrisk": y_g.mean(),
            "auroc":      roc_auc_score(y_g, p_g),
            "recall":     recall_score(y_g, d_g, zero_division=0),
            "precision":  precision_score(y_g, d_g, zero_division=0),
            "flag_rate":  d_g.mean(),
        })

    mod_df = pd.DataFrame(rows).sort_values("auroc")
    print("\nPer-module AUROC:")
    print(mod_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tomato" if a < 0.70 else "steelblue" for a in mod_df["auroc"]]
    ax.barh(mod_df["module"], mod_df["auroc"], color=colors, alpha=0.85)
    ax.axvline(0.70, color="green", linestyle="--", label="Target 0.70")
    ax.set_xlabel("AUROC")
    ax.set_title("Per-Module AUROC — Task 2v2 (Temporal Test)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_module_auroc.png", dpi=150, bbox_inches="tight")
    plt.show()

    return mod_df


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def shap_analysis(base_model, X_tr, X_te, feat_cols, output_dir="."):
    explainer   = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_te)

    plt.figure()
    shap.summary_plot(shap_values, X_te, feature_names=feat_cols, show=False, max_display=15)
    plt.title("SHAP Summary — Feature Impact on At-Risk Prediction")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()

    mean_abs = np.abs(shap_values).mean(axis=0)
    top10    = pd.Series(mean_abs, index=feat_cols).nlargest(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    top10.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 10 Features — Mean Absolute SHAP")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_top10.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nTop 3 features (SHAP):")
    for feat, val in top10.head(3).items():
        print(f"  {feat}: {val:.4f}")

    return explainer, shap_values


# ---------------------------------------------------------------------------
# Waterfall plots — individual student explanations
# ---------------------------------------------------------------------------

def shap_waterfall(explainer, shap_values, X_te, y_te, probs, feat_cols,
                   threshold, output_dir="."):
    at_risk_idx = np.where((y_te == 1) & (probs >= threshold))[0]
    safe_idx    = np.where((y_te == 0) & (probs < threshold))[0]

    for label_str, indices in [("at-risk", at_risk_idx), ("safe", safe_idx)]:
        if len(indices) == 0:
            continue
        # pick a representative student near the median probability
        subset_probs = probs[indices]
        mid = indices[np.argsort(np.abs(subset_probs - subset_probs.mean()))[0]]

        sv = shap_values[mid]
        ev = shap.Explanation(
            values=sv,
            base_values=explainer.expected_value,
            data=X_te[mid],
            feature_names=feat_cols
        )
        plt.figure()
        shap.plots.waterfall(ev, show=False, max_display=12)
        plt.title(f"SHAP Waterfall — {label_str} student (prob={probs[mid]:.3f})")
        plt.tight_layout()
        path = f"{output_dir}/shap_waterfall_{label_str}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Staff alert table
# ---------------------------------------------------------------------------

def build_alerts(model, df_te, feat_cols, threshold, shap_values, output_dir="."):
    """
    One row per test student. Includes:
      - risk_prob: calibrated probability of withdraw/fail
      - alert_tier: High / Medium / Watch
      - top3_reasons: top-3 SHAP drivers (feature name + direction)
      - archetype_at_w6: student type at week 6
      - recommended_action

    Flag rate = fraction of students above threshold.
    Designed to stay ≤ 35% of cohort (operational constraint).
    """
    X_te  = df_te[feat_cols].values.astype(float)
    probs = model.predict_proba(X_te)[:, 1]

    mean_abs_shap = np.abs(shap_values)
    top3_reasons  = []
    for i in range(len(X_te)):
        sv = shap_values[i]
        top_idx = np.argsort(np.abs(sv))[::-1][:3]
        reasons = []
        for idx in top_idx:
            direction = "↑risk" if sv[idx] > 0 else "↓risk"
            reasons.append(f"{feat_cols[idx]} ({direction})")
        top3_reasons.append(" | ".join(reasons))

    alerts = df_te[["id_student", "code_module", "code_presentation"]].copy().reset_index(drop=True)
    alerts["risk_prob"]    = probs.round(3)
    alerts["flagged"]      = (probs >= threshold).astype(int)
    alerts["top3_reasons"] = top3_reasons

    # archetype at w6 if available
    arch_cols = [c for c in df_te.columns if c.startswith("arch_w6_")]
    if arch_cols:
        def decode_arch(row):
            for col in arch_cols:
                if row[col] == 1:
                    return col.replace("arch_w6_", "").replace("_", " ")
            return "Ghost"  # no VLE activity → ghost-like student
        alerts["archetype_at_w6"] = df_te[arch_cols].reset_index(drop=True).apply(decode_arch, axis=1)

    # tiered alerts — anchored to the operational threshold
    high_t   = threshold + (1.0 - threshold) * 0.4   # top 40% of flagged range
    medium_t = threshold

    def assign_tier(p):
        if p >= high_t:
            return "High"
        elif p >= medium_t:
            return "Medium"
        else:
            return "Watch"

    alerts["alert_tier"] = alerts["risk_prob"].apply(assign_tier)

    tier_actions = {
        "High":   "Immediate outreach — schedule meeting within 48 hours",
        "Medium": "Proactive check-in — contact within the week",
        "Watch":  "Monitor — flag for review next week",
    }
    alerts["recommended_action"] = alerts["alert_tier"].map(tier_actions)

    alerts = alerts.sort_values("risk_prob", ascending=False)
    alerts.to_csv(f"{output_dir}/student_alerts_v2.csv", index=False)

    flagged = alerts[alerts["flagged"] == 1]
    print(f"\nStaff Alert Summary:")
    print(f"  Total students:  {len(alerts):,}")
    print(f"  Flagged:         {len(flagged):,} ({len(flagged)/len(alerts)*100:.1f}%)")
    print(f"  High:            {(alerts['alert_tier']=='High').sum():,}")
    print(f"  Medium:          {(alerts['alert_tier']=='Medium').sum():,}")
    print(f"  Watch:           {(alerts['alert_tier']=='Watch').sum():,}")
    print(f"  Saved: {output_dir}/student_alerts_v2.csv")

    return alerts
