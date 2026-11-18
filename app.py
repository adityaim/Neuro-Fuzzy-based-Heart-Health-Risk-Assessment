"""
app.py
------
Streamlit demonstration of the Neuro-Fuzzy Heart Health Risk Assessment
system. Run `python train.py` once first to generate artifacts/, results/
and figures/, then launch with:

    streamlit run app.py

Two tabs:
  1. "Explanation & Metrics" - methodology walkthrough + the report's
     headline figures and tables, loaded from disk.
  2. "Live Demo" - enter a patient's clinical attributes and see the
     Fuzzy-Only, Logistic Regression, Baseline ANN and Neuro-Fuzzy
     predictions side by side, plus the rule firing strengths that drove
     the Neuro-Fuzzy prediction.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from fuzzy_system import RULES, MF_PARAMS
from data_pipeline import NUMERIC_COLS, CAT_COLS

st.set_page_config(page_title="Neuro-Fuzzy Heart Health Risk Assessment", layout="wide")

ARTIFACT_PATH = "artifacts/bundle.joblib"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
ASSETS_DIR = "assets"


@st.cache_resource
def load_bundle():
    if not os.path.exists(ARTIFACT_PATH):
        return None
    return joblib.load(ARTIFACT_PATH)


@st.cache_data
def load_results():
    data = {}
    for name in ["test_results.csv", "cv_summary.csv"]:
        p = os.path.join(RESULTS_DIR, name)
        if os.path.exists(p):
            data[name] = pd.read_csv(p)
    summary_p = os.path.join(RESULTS_DIR, "summary.json")
    data["summary"] = json.load(open(summary_p)) if os.path.exists(summary_p) else {}
    return data


bundle = load_bundle()
results = load_results()

st.title("🫀 Neuro-Fuzzy Heart Health Risk Assessment")
st.caption("A hybrid fuzzy-logic + neural-network system for illustrative, non-diagnostic heart-disease risk screening.")

tab1, tab2 = st.tabs(["📘 Explanation & Metrics", "🩺 Live Demo"])

# =============================================================================
# TAB 1 — EXPLANATION & METRICS
# =============================================================================
with tab1:
    st.warning(
        "⚠️ **Illustrative risk categories, not a clinical diagnosis.** "
        "This is a course project built on a public research dataset. The membership "
        "functions and fuzzy rules were set from general clinical reference ranges and were "
        "not reviewed by a cardiologist. Do not use this tool to make real medical decisions."
    )

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.header("How it works")
        st.markdown(
            """
The system follows a four-stage pipeline:

1. **Fuzzification** — Five numeric attributes (Age, RestingBP, Cholesterol,
   MaxHR, Oldpeak) are each converted into *Low / Medium / High* membership
   degrees using triangular membership functions grounded in standard
   clinical reference ranges.
2. **Fuzzy Rule Base** — Ten clinically-motivated Sugeno-style rules
   (e.g. *"IF Age is High AND Oldpeak is High THEN Risk is High"*) combine
   these memberships via product T-norms into rule firing strengths and a
   zero-training **Fuzzy Risk Index**.
3. **Neural Learning Component** — A feedforward network (2 hidden layers,
   16→8 ReLU units), trained by back-propagation, learns from the fuzzified
   memberships and rule firing strengths (NOT the raw numbers) to produce a
   final heart-disease probability.
4. **Risk Assessment** — The probability is reported directly, and banded
   into an illustrative **Low / Moderate / High** risk category.
            """
        )
    with col2:
        flowchart_path = os.path.join(ASSETS_DIR, "flowchart.png")
        if os.path.exists(flowchart_path):
            st.image(flowchart_path, caption="Proposed Neuro-Fuzzy Workflow", width="stretch")

    st.divider()
    st.header("Fuzzy Rule Base")
    rule_rows = [{"#": f"R{i+1}", "Rule": name.split(": ", 1)[1], "Consequent (risk weight)": c}
                 for i, (name, _, c) in enumerate(RULES)]
    st.dataframe(pd.DataFrame(rule_rows), width="stretch", hide_index=True)

    mf_path = os.path.join(FIGURES_DIR, "fig_membership_functions.png")
    if os.path.exists(mf_path):
        st.image(mf_path, caption="Triangular membership functions for the five fuzzified attributes", width="stretch")

    st.divider()
    st.header("Experimental Results")
    if bundle is None:
        st.error("No trained artifacts found. Run `python train.py` first, then restart this app.")
    else:
        s = results.get("summary", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Patients (total)", s.get("n_total", "—"))
        m2.metric("Train / Test split", f"{s.get('n_train','—')} / {s.get('n_test','—')}")
        m3.metric("Cholesterol zeros imputed", s.get("n_chol_imputed", "—"))
        m4.metric("RestingBP zeros imputed", s.get("n_bp_imputed", "—"))

        st.subheader("Test-set performance (held-out, 184 patients)")
        if "test_results.csv" in results:
            st.dataframe(results["test_results.csv"], width="stretch", hide_index=True)
        st.caption(
            "Logistic Regression is the strongest single-split performer here; the Neuro-Fuzzy "
            "model ties the Baseline ANN on accuracy but gets a higher recall/F1. See the 5-fold "
            "CV below for a more reliable comparison."
        )

        st.subheader("5-fold cross-validation summary (mean ± std)")
        if "cv_summary.csv" in results:
            st.dataframe(results["cv_summary.csv"], width="stretch", hide_index=True)
        st.caption(
            "Across 5 folds, the Neuro-Fuzzy model shows a small, consistent edge over the plain "
            "Baseline ANN in accuracy and F1 — modest, but reproducible."
        )

        c1, c2 = st.columns(2)
        with c1:
            p = os.path.join(FIGURES_DIR, "fig_training_curves.png")
            if os.path.exists(p):
                st.image(p, caption="Validation loss/accuracy vs epoch (early-stopping point marked)", width="stretch")
        with c2:
            p = os.path.join(FIGURES_DIR, "fig_roc_curves.png")
            if os.path.exists(p):
                st.image(p, caption="ROC curves — test set", width="stretch")

        p = os.path.join(FIGURES_DIR, "fig_confusion_matrices.png")
        if os.path.exists(p):
            st.image(p, caption="Confusion matrices — Fuzzy-Only, Baseline ANN, Neuro-Fuzzy", width="stretch")

# =============================================================================
# TAB 2 — LIVE DEMO
# =============================================================================
with tab2:
    st.warning(
        "⚠️ **For demonstration only — not a medical device.** Illustrative risk "
        "categories, not a clinical diagnosis. If you have real health concerns, consult a doctor."
    )

    if bundle is None:
        st.error("No trained artifacts found. Run `python train.py` first, then restart this app.")
        st.stop()

    fb = bundle["feature_builder"]
    logreg = bundle["logreg"]
    mlp_base = bundle["baseline_ann"]
    mlp_nf = bundle["neurofuzzy_ann"]

    st.header("Enter patient attributes")
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age (years)", 20, 85, 54)
            sex = st.selectbox("Sex", ["M", "F"])
            chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"],
                                       help="ASY: Asymptomatic, ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina")
            resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
        with c2:
            cholesterol = st.slider("Serum Cholesterol (mg/dl)", 100, 450, 240)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
            max_hr = st.slider("Maximum Heart Rate Achieved", 60, 202, 140)
        with c3:
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
            oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
            st_slope = st.selectbox("ST Slope (peak exercise)", ["Up", "Flat", "Down"])
        submitted = st.form_submit_button("🔍 Assess Risk", width="stretch")

    if submitted:
        record = {
            "Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
            "Cholesterol": cholesterol, "FastingBS": 1 if fasting_bs == "Yes" else 0,
            "RestingECG": resting_ecg, "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak, "ST_Slope": st_slope,
        }

        x_base = fb.transform_baseline_record(record).reshape(1, -1)
        x_nf, fri, firings, fuzzy_vals = fb.transform_neurofuzzy_record(record)
        x_nf = x_nf.reshape(1, -1)

        p_logreg = float(logreg.predict_proba(x_base)[0, 1])
        p_base = float(mlp_base.predict_proba(x_base)[0, 1])
        p_nf = float(mlp_nf.predict_proba(x_nf)[0, 1])

        def band(p):
            if p < 0.33:
                return "Low", "🟢"
            elif p < 0.66:
                return "Moderate", "🟡"
            return "High", "🔴"

        band_name, band_icon = band(p_nf)

        st.divider()
        st.header("Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Neuro-Fuzzy probability", f"{p_nf:.1%}")
        r2.metric("Predicted risk band", f"{band_icon} {band_name}")
        r3.metric("Fuzzy Risk Index (rule-only)", f"{fri:.2f}")

        st.subheader("Comparison across all four models")
        comp_df = pd.DataFrame([
            {"Model": "Fuzzy-Only (zero-training rule base)", "Probability / Index": round(fri, 3)},
            {"Model": "Logistic Regression", "Probability / Index": round(p_logreg, 3)},
            {"Model": "Baseline ANN (no fuzzy layer)", "Probability / Index": round(p_base, 3)},
            {"Model": "Neuro-Fuzzy (proposed)", "Probability / Index": round(p_nf, 3)},
        ])
        st.dataframe(comp_df, width="stretch", hide_index=True)

        st.subheader("Which rules fired for this patient?")
        fire_df = pd.DataFrame({"Rule": [n.split(": ", 1)[1] for n, _, _ in RULES],
                                 "Firing strength": list(firings.values())}).sort_values("Firing strength", ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#C44E52" if v >= 0.5 else "#4C72B0" for v in fire_df["Firing strength"]]
        ax.barh(fire_df["Rule"], fire_df["Firing strength"], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Firing strength")
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Show fuzzified membership degrees"):
            st.dataframe(pd.DataFrame([fuzzy_vals]).T.rename(columns={0: "Membership degree"}), width="stretch")

        st.info(
            "💡 This breakdown is the point of the neuro-fuzzy design: unlike a plain neural "
            "network's weights, these named rules can be read and sanity-checked by a human."
        )
