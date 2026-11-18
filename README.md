# Neuro-Fuzzy Heart Health Risk Assessment — Reference Implementation

This is the Python implementation backing the project report *"Neuro-Fuzzy
based Heart Health Risk Assessment"*. It independently reproduces every
number in the report's Results chapter, and ships a Streamlit app for
interactive demonstration.

Live Demonstration can be viewed at: https://neuro-fuzzy-based-heart-health-risk-assessment.streamlit.app/

## Contents

| File | Purpose |
|---|---|
| `fuzzy_system.py` | Membership functions, the 10-rule fuzzy base, Fuzzy Risk Index |
| `data_pipeline.py` | Data cleaning (zero-value imputation) and the `FeatureBuilder` (shared encoder/scaler for training *and* single-patient inference) |
| `models.py` | Model definitions + the early-stopping training loop |
| `train.py` | End-to-end training/evaluation script — run this first |
| `app.py` | Streamlit demo (2 tabs: Explanation & Metrics, Live Demo) |
| `heart.csv` | The Heart Failure Prediction dataset (918 records) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train and evaluate (reproduces the report's numbers)

```bash
python train.py
```

This will:
- clean the dataset (median-imputes 172 invalid zero-Cholesterol and 1 invalid zero-RestingBP readings),
- fuzzify the five numeric attributes and compute the 10-rule Fuzzy Risk Index,
- train the Baseline ANN and Neuro-Fuzzy ANN with early stopping (best epochs printed to console),
- evaluate all four models (Fuzzy-Only, Logistic Regression, Baseline ANN, Neuro-Fuzzy) on a held-out test set and via 5-fold cross-validation,
- save trained artifacts to `artifacts/bundle.joblib`, metrics to `results/*.csv`, and figures to `figures/*.png`.

Expected console output (random_state = 42 throughout, matches report Table 7 / Table 8):

```
Baseline ANN   : accuracy 0.837  f1 0.856  auc 0.917   (best epoch 9)
Neuro-Fuzzy    : accuracy 0.837  f1 0.859  auc 0.903   (best epoch 18)
Logistic Reg.  : accuracy 0.886  f1 0.899  auc 0.932
Fuzzy-Only     : accuracy 0.826  f1 0.857  auc 0.846

5-fold CV:
Baseline ANN   : accuracy 0.846±0.026  f1 0.863±0.021
Neuro-Fuzzy    : accuracy 0.850±0.026  f1 0.867±0.020
```

## 2. Launch the demo

```bash
streamlit run app.py
```

Opens a two-tab app:

- **Explanation & Metrics** — methodology walkthrough, the fuzzy rule table,
  membership-function plots, and all of the report's performance tables and
  figures, loaded live from `results/` and `figures/`.
- **Live Demo** — enter a patient's 11 clinical attributes and get, side by
  side: the Fuzzy-Only index, Logistic Regression, Baseline ANN, and
  Neuro-Fuzzy probabilities, the predicted Low/Moderate/High band, and a bar
  chart of exactly which of the 10 fuzzy rules fired and how strongly —
  the interpretability feature that is the whole point of the neuro-fuzzy
  design (see report, Section 5.7–5.9).

> ⚠️ **Disclaimer carried over from the report:** the membership-function
> breakpoints and rule base were set from general clinical reference ranges,
> not reviewed by a cardiologist. All risk outputs are illustrative only and
> are not a validated clinical screening tool.

## Design notes

- `fuzzy_system.py` and `data_pipeline.py` are written so that the **exact
  same functions** fuzzify a full training DataFrame and a single
  interactively-entered patient — there is no separate "demo-only" copy of
  the fuzzification/rule logic that could silently drift from what was
  trained on.
- `FeatureBuilder` persists the fitted `OneHotEncoder` and `StandardScaler`
  inside `artifacts/bundle.joblib`, so the Streamlit app never re-fits
  preprocessing on the fly — it uses precisely the transformation the models
  were trained with.
- All random state is fixed (`random_state = 42`) for reproducibility.
