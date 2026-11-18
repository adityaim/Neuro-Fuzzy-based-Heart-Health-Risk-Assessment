"""
data_pipeline.py
-----------------
Data loading, cleaning, and feature construction for the Neuro-Fuzzy Heart
Health Risk Assessment system (mirrors Chapter 3 / Chapter 4 of the report).

The FeatureBuilder class fits a OneHotEncoder and a StandardScaler once on
the training data and reuses them both for building the two model input
spaces (Baseline-ANN and Neuro-Fuzzy) at training time AND for transforming
a single, interactively-entered patient at demo time -- guaranteeing that
training and inference always see identically-shaped, identically-scaled
features.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from fuzzy_system import (
    NUMERIC_COLS, CAT_COLS, fuzzify_dataframe, fuzzify_record,
    compute_rule_firing_dataframe, compute_rule_firing_record,
)

ALL_INPUT_COLS = NUMERIC_COLS + CAT_COLS + ["FastingBS"]


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute the physiologically-impossible zero readings in
    Cholesterol and RestingBP (see report, Section 3.3)."""
    df = df.copy()
    chol_median = df.loc[df["Cholesterol"] > 0, "Cholesterol"].median()
    bp_median = df.loc[df["RestingBP"] > 0, "RestingBP"].median()
    n_chol = int((df["Cholesterol"] == 0).sum())
    n_bp = int((df["RestingBP"] == 0).sum())
    df.loc[df["Cholesterol"] == 0, "Cholesterol"] = chol_median
    df.loc[df["RestingBP"] == 0, "RestingBP"] = bp_median
    meta = {"chol_median": float(chol_median), "bp_median": float(bp_median),
            "n_chol_imputed": n_chol, "n_bp_imputed": n_bp}
    return df, meta


class FeatureBuilder:
    """Fits categorical/numeric encoders once; builds both the Baseline-ANN
    feature matrix (raw scaled numeric + categorical, 18-dim) and the
    Neuro-Fuzzy feature matrix (fuzzified memberships + rule firing
    strengths + categorical, 38-dim), for either a full DataFrame or a
    single patient record."""

    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False, drop="if_binary")
        self.scaler = StandardScaler()
        self.cat_feature_names_ = None
        self.fitted_ = False

    def fit(self, df: pd.DataFrame):
        self.ohe.fit(df[CAT_COLS])
        self.cat_feature_names_ = list(self.ohe.get_feature_names_out(CAT_COLS))
        self.scaler.fit(df[NUMERIC_COLS])
        self.fitted_ = True
        return self

    # -- whole-dataframe transforms (training) --------------------------------
    def transform_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled = pd.DataFrame(self.scaler.transform(df[NUMERIC_COLS]), columns=NUMERIC_COLS, index=df.index)
        cat = pd.DataFrame(self.ohe.transform(df[CAT_COLS]), columns=self.cat_feature_names_, index=df.index)
        fbs = df[["FastingBS"]].astype(float)
        return pd.concat([scaled, fbs, cat], axis=1)

    def transform_neurofuzzy(self, df: pd.DataFrame):
        fuzzy_df = fuzzify_dataframe(df)
        firing_df, fri = compute_rule_firing_dataframe(df)
        cat = pd.DataFrame(self.ohe.transform(df[CAT_COLS]), columns=self.cat_feature_names_, index=df.index)
        fbs = df[["FastingBS"]].astype(float)
        X = pd.concat([fuzzy_df, firing_df, fbs, cat], axis=1)
        return X, fri

    # -- single-record transforms (interactive demo) --------------------------
    def transform_baseline_record(self, record: dict) -> np.ndarray:
        numeric_df = pd.DataFrame([{c: record[c] for c in NUMERIC_COLS}])
        scaled = self.scaler.transform(numeric_df)[0]
        cat_df = pd.DataFrame([{c: record[c] for c in CAT_COLS}])
        cat = self.ohe.transform(cat_df)[0]
        fbs = np.array([float(record["FastingBS"])])
        return np.concatenate([scaled, fbs, cat])

    def transform_neurofuzzy_record(self, record: dict):
        firings, fri, fuzzy_vals = compute_rule_firing_record(record)
        fuzzy_vec = np.array(list(fuzzy_vals.values()))
        firing_vec = np.array(list(firings.values()))
        cat_df = pd.DataFrame([{c: record[c] for c in CAT_COLS}])
        cat = self.ohe.transform(cat_df)[0]
        fbs = np.array([float(record["FastingBS"])])
        X = np.concatenate([fuzzy_vec, firing_vec, fbs, cat])
        return X, fri, firings, fuzzy_vals
