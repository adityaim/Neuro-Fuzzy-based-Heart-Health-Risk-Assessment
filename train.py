"""
train.py
--------
End-to-end training + evaluation script for the Neuro-Fuzzy Heart Health
Risk Assessment system. Running this script:

  1. Loads and cleans heart.csv (median-imputes invalid zero readings).
  2. Fits the FeatureBuilder (OneHotEncoder + StandardScaler).
  3. Trains four models under an identical, honest evaluation protocol:
       - Fuzzy-Only       (zero-training rule base)
       - Logistic Regression (18-dim raw features)
       - Baseline ANN     (18-dim raw features, no fuzzy front end)
       - Neuro-Fuzzy ANN  (38-dim fuzzified + rule-firing features)
  4. Evaluates all four on a held-out test set, plus 5-fold CV for the two
     neural models.
  5. Saves: trained artifacts (artifacts/bundle.joblib), metrics
     (results/*.csv), and the figures used in the report (figures/*.png).

This reproduces the numbers reported in Chapter 5 of the project report
(random_state = 42 throughout, same architecture/hyperparameters).

Usage:
    python train.py
"""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix)

from fuzzy_system import RULES
from data_pipeline import load_raw, clean_data, FeatureBuilder, NUMERIC_COLS
from models import train_with_early_stopping, make_logreg, RANDOM_STATE

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 150


def evaluate(name, ytrue, yprob, thresh=0.5):
    ypred = (yprob >= thresh).astype(int)
    return {"model": name, "accuracy": accuracy_score(ytrue, ypred),
            "precision": precision_score(ytrue, ypred), "recall": recall_score(ytrue, ypred),
            "f1": f1_score(ytrue, ypred), "roc_auc": roc_auc_score(ytrue, yprob)}


def main():
    df_raw = load_raw("heart.csv")
    df, clean_meta = clean_data(df_raw)
    df.to_csv("heart_clean.csv", index=False)
    print("Data-cleaning summary:", clean_meta)

    fb = FeatureBuilder().fit(df)
    X_baseline_full = fb.transform_baseline(df)
    X_neurofuzzy_full, fri_full = fb.transform_neurofuzzy(df)
    y = df["HeartDisease"].values

    Xb_tr, Xb_te, Xnf_tr, Xnf_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_baseline_full.values, X_neurofuzzy_full.values, y, df.index.values,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    Xb_tr2, Xb_val, Xnf_tr2, Xnf_val, y_tr2, y_val = train_test_split(
        Xb_tr, Xnf_tr, y_tr, test_size=0.15, random_state=RANDOM_STATE, stratify=y_tr)

    print("Training Baseline ANN with early stopping...")
    mlp_base, be_base, hist_base = train_with_early_stopping(Xb_tr2, y_tr2, Xb_val, y_val)
    print(f"  best epoch = {be_base}")

    print("Training Neuro-Fuzzy ANN with early stopping...")
    mlp_nf, be_nf, hist_nf = train_with_early_stopping(Xnf_tr2, y_tr2, Xnf_val, y_val)
    print(f"  best epoch = {be_nf}")

    logreg = make_logreg().fit(Xb_tr, y_tr)

    results = [
        evaluate("Fuzzy-Only (zero-training rule base)", y_te, fri_full[idx_te]),
        evaluate("Logistic Regression (baseline)", y_te, logreg.predict_proba(Xb_te)[:, 1]),
        evaluate("Baseline ANN (raw features, no fuzzy layer)", y_te, mlp_base.predict_proba(Xb_te)[:, 1]),
        evaluate("Neuro-Fuzzy Model (fuzzified + rules + ANN)", y_te, mlp_nf.predict_proba(Xnf_te)[:, 1]),
    ]
    res_df = pd.DataFrame(results).round(4)
    print("\n=== TEST SET RESULTS ===")
    print(res_df.to_string(index=False))
    res_df.to_csv("results/test_results.csv", index=False)

    # 5-fold CV for the two neural models
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for fold, (tr_i, te_i) in enumerate(skf.split(X_baseline_full.values, y)):
        for name, X in [("Baseline ANN", X_baseline_full.values), ("Neuro-Fuzzy", X_neurofuzzy_full.values)]:
            from sklearn.neural_network import MLPClassifier
            m = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
                               alpha=1e-3, max_iter=400, random_state=RANDOM_STATE, learning_rate_init=0.01,
                               early_stopping=True, n_iter_no_change=15, validation_fraction=0.15)
            m.fit(X[tr_i], y[tr_i])
            p = m.predict_proba(X[te_i])[:, 1]
            cv_rows.append({"fold": fold + 1, "model": name, "accuracy": accuracy_score(y[te_i], p > 0.5),
                             "f1": f1_score(y[te_i], p > 0.5), "roc_auc": roc_auc_score(y[te_i], p)})
    cv_df = pd.DataFrame(cv_rows)
    cv_summary = cv_df.groupby("model")[["accuracy", "f1", "roc_auc"]].agg(["mean", "std"]).round(4)
    print("\n=== 5-FOLD CROSS-VALIDATION SUMMARY ===")
    print(cv_summary)
    cv_df.to_csv("results/cv_results.csv", index=False)
    cv_summary.to_csv("results/cv_summary.csv")

    # ---- figures (same as used in the report) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(hist_base["val_loss"], label="Baseline ANN", color="#4C72B0")
    axes[0].plot(hist_nf["val_loss"], label="Neuro-Fuzzy", color="#C44E52")
    axes[0].axvline(be_base, color="#4C72B0", ls="--", lw=1, alpha=0.6)
    axes[0].axvline(be_nf, color="#C44E52", ls="--", lw=1, alpha=0.6)
    axes[0].set_title("Validation Loss vs Epoch (dashed = early-stop point)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Binary Cross-Entropy"); axes[0].legend()
    axes[1].plot(hist_base["val_acc"], label="Baseline ANN", color="#4C72B0")
    axes[1].plot(hist_nf["val_acc"], label="Neuro-Fuzzy", color="#C44E52")
    axes[1].axvline(be_base, color="#4C72B0", ls="--", lw=1, alpha=0.6)
    axes[1].axvline(be_nf, color="#C44E52", ls="--", lw=1, alpha=0.6)
    axes[1].set_title("Validation Accuracy vs Epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
    plt.tight_layout(); plt.savefig("figures/fig_training_curves.png"); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, (name, yprob) in zip(axes, [("Fuzzy-Only", fri_full[idx_te]),
                                          ("Baseline ANN", mlp_base.predict_proba(Xb_te)[:, 1]),
                                          ("Neuro-Fuzzy", mlp_nf.predict_proba(Xnf_te)[:, 1])]):
        cm = confusion_matrix(y_te, (yprob >= 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Normal", "Disease"], yticklabels=["Normal", "Disease"])
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig("figures/fig_confusion_matrices.png"); plt.close()

    fig, ax = plt.subplots(figsize=(6, 5.3))
    for name, yprob, color in [("Fuzzy-Only", fri_full[idx_te], "#55A868"),
                                 ("Logistic Regression", logreg.predict_proba(Xb_te)[:, 1], "#8172B2"),
                                 ("Baseline ANN", mlp_base.predict_proba(Xb_te)[:, 1], "#4C72B0"),
                                 ("Neuro-Fuzzy", mlp_nf.predict_proba(Xnf_te)[:, 1], "#C44E52")]:
        fpr, tpr, _ = roc_curve(y_te, yprob); auc = roc_auc_score(y_te, yprob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Test Set"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig("figures/fig_roc_curves.png"); plt.close()

    # membership function plot
    from fuzzy_system import MF_PARAMS, triangular_mf
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    xr = {"Age": np.linspace(20, 85, 300), "RestingBP": np.linspace(75, 205, 300),
          "Cholesterol": np.linspace(95, 410, 300), "MaxHR": np.linspace(55, 205, 300),
          "Oldpeak": np.linspace(-3, 6.6, 300)}
    colors = {"Low": "#4C72B0", "Medium": "#DD8452", "High": "#C44E52"}
    for i, col in enumerate(NUMERIC_COLS):
        for term, (a, b, c) in MF_PARAMS[col].items():
            axes[i].plot(xr[col], triangular_mf(xr[col], a, b, c), label=term, color=colors[term], lw=2)
        axes[i].set_title(f"{col} - Membership Functions"); axes[i].legend(fontsize=8)
    axes[-1].axis("off")
    plt.tight_layout(); plt.savefig("figures/fig_membership_functions.png"); plt.close()

    # ---- save artifact bundle for the Streamlit app ----
    bundle = {
        "feature_builder": fb,
        "logreg": logreg,
        "baseline_ann": mlp_base,
        "neurofuzzy_ann": mlp_nf,
        "best_epoch_baseline": be_base,
        "best_epoch_neurofuzzy": be_nf,
        "clean_meta": clean_meta,
        "test_indices": idx_te,
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    }
    joblib.dump(bundle, "artifacts/bundle.joblib")

    summary = {
        "n_total": int(len(df)), "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "class_balance": {"disease": int(y.sum()), "normal": int(len(y) - y.sum())},
        "best_epoch_baseline": int(be_base), "best_epoch_neurofuzzy": int(be_nf),
        **clean_meta,
    }
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll artifacts saved under artifacts/, results/, figures/.")
    print("DONE. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
