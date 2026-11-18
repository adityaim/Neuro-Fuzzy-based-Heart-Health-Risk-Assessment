"""
fuzzy_system.py
----------------
Core fuzzy-logic layer of the Neuro-Fuzzy Heart Health Risk Assessment system.

This module implements exactly the fuzzification scheme and the 10-rule
Sugeno-style fuzzy inference system described in Chapter 3 of the project
report ("Neuro-Fuzzy based Heart Health Risk Assessment"). It is written to
work identically on:
  * a full pandas DataFrame (vectorised, used for training), and
  * a single patient's attributes given as a plain dict (used by the
    Streamlit demo for one-off, interactive predictions).

Both usages share the same membership-function parameters and the same rule
definitions, so training and demo inference can never silently drift apart.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CAT_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
LINGUISTIC_TERMS = ["Low", "Medium", "High"]

# ---------------------------------------------------------------------------
# Triangular membership-function parameters (a, b, c), identical to Table 4
# of the report. Breakpoints are grounded in standard cardiology reference
# ranges (see Chapter 3, Section 3.4 of the report).
# ---------------------------------------------------------------------------
MF_PARAMS = {
    "Age":         {"Low": (20, 30, 46),   "Medium": (36, 52, 62),   "High": (52, 68, 82)},
    "RestingBP":   {"Low": (80, 100, 122), "Medium": (112, 130, 150), "High": (138, 160, 200)},
    "Cholesterol": {"Low": (100, 150, 205), "Medium": (185, 225, 260), "High": (245, 300, 400)},
    "MaxHR":       {"Low": (60, 90, 112),  "Medium": (100, 130, 152), "High": (140, 170, 202)},
    "Oldpeak":     {"Low": (-3, 0, 0.6),   "Medium": (0.3, 1.1, 2.1), "High": (1.6, 3.0, 6.5)},
}


def triangular_mf(x, a, b, c):
    """Triangular membership function mu(x; a, b, c), works on scalars or
    numpy arrays / pandas Series alike."""
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = np.atleast_1d(x)
    y = np.zeros_like(x)
    left = (x > a) & (x <= b)
    right = (x > b) & (x < c)
    peak = (x == b)
    y[left] = (x[left] - a) / (b - a + 1e-9)
    y[right] = (c - x[right]) / (c - b + 1e-9)
    y[peak] = 1.0
    y = np.clip(y, 0, 1)
    return float(y[0]) if scalar_input else y


def fuzzify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised fuzzification of all NUMERIC_COLS for an entire DataFrame.
    Returns a DataFrame with 15 columns: '<Attribute>_<Term>' in [0, 1]."""
    out = {}
    for col in NUMERIC_COLS:
        for term, (a, b, c) in MF_PARAMS[col].items():
            out[f"{col}_{term}"] = triangular_mf(df[col].values, a, b, c)
    return pd.DataFrame(out, index=df.index)


def fuzzify_record(record: dict) -> dict:
    """Fuzzify a single patient's numeric attributes (as a flat dict), e.g.
    {'Age': 58, 'RestingBP': 140, ...} -> {'Age_Low': 0.0, 'Age_Medium': 0.4, ...}"""
    out = {}
    for col in NUMERIC_COLS:
        for term, (a, b, c) in MF_PARAMS[col].items():
            out[f"{col}_{term}"] = triangular_mf(record[col], a, b, c)
    return out


# ---------------------------------------------------------------------------
# Fuzzy rule base (Table 5 of the report): 10 clinically-motivated rules.
# Each rule is (name, firing_fn, consequent). firing_fn takes a "context"
# dict that contains BOTH the 15 fuzzy membership degrees (keys like
# 'Age_High') AND crisp 0/1 indicators for the relevant categorical
# attributes (keys like 'ExerciseAngina_Y', 'ST_Slope_Flat', ...). Because
# plain multiplication broadcasts correctly over both python floats and
# numpy arrays, the exact same rule definitions serve vectorised training
# and single-patient demo inference without any code duplication.
# ---------------------------------------------------------------------------
RULES = [
    ("R1: Age=High & Oldpeak=High",
     lambda c: c["Age_High"] * c["Oldpeak_High"], 0.95),
    ("R2: ExerciseAngina=Y & MaxHR=Low",
     lambda c: c["ExerciseAngina_Y"] * c["MaxHR_Low"], 0.90),
    ("R3: ChestPain=ASY & Oldpeak=Medium",
     lambda c: c["ChestPainType_ASY"] * c["Oldpeak_Medium"], 0.80),
    ("R4: ST_Slope=Flat & Oldpeak=High",
     lambda c: c["ST_Slope_Flat"] * c["Oldpeak_High"], 0.95),
    ("R5: ST_Slope=Down",
     lambda c: c["ST_Slope_Down"], 0.85),
    ("R6: Age=High & Chol=High & BP=High",
     lambda c: c["Age_High"] * c["Cholesterol_High"] * c["RestingBP_High"], 0.75),
    ("R7: FastingBS=1 & Age=High",
     lambda c: c["FastingBS"] * c["Age_High"], 0.70),
    ("R8: Age=Low & MaxHR=High & Oldpeak=Low",
     lambda c: c["Age_Low"] * c["MaxHR_High"] * c["Oldpeak_Low"], 0.05),
    ("R9: ST_Slope=Up & Oldpeak=Low & Angina=N",
     lambda c: c["ST_Slope_Up"] * c["Oldpeak_Low"] * (1 - c["ExerciseAngina_Y"]), 0.05),
    ("R10: ChestPain=ATA & MaxHR=High",
     lambda c: c["ChestPainType_ATA"] * c["MaxHR_High"], 0.10),
]


def _crisp_indicators_from_df(df: pd.DataFrame) -> dict:
    return {
        "ExerciseAngina_Y": df["ExerciseAngina"].eq("Y").astype(float).values,
        "ChestPainType_ASY": df["ChestPainType"].eq("ASY").astype(float).values,
        "ChestPainType_ATA": df["ChestPainType"].eq("ATA").astype(float).values,
        "ST_Slope_Flat": df["ST_Slope"].eq("Flat").astype(float).values,
        "ST_Slope_Down": df["ST_Slope"].eq("Down").astype(float).values,
        "ST_Slope_Up": df["ST_Slope"].eq("Up").astype(float).values,
        "FastingBS": df["FastingBS"].astype(float).values,
    }


def _crisp_indicators_from_record(record: dict) -> dict:
    return {
        "ExerciseAngina_Y": float(record["ExerciseAngina"] == "Y"),
        "ChestPainType_ASY": float(record["ChestPainType"] == "ASY"),
        "ChestPainType_ATA": float(record["ChestPainType"] == "ATA"),
        "ST_Slope_Flat": float(record["ST_Slope"] == "Flat"),
        "ST_Slope_Down": float(record["ST_Slope"] == "Down"),
        "ST_Slope_Up": float(record["ST_Slope"] == "Up"),
        "FastingBS": float(record["FastingBS"]),
    }


def compute_rule_firing_dataframe(df: pd.DataFrame):
    """Returns (firing_df, fuzzy_risk_index array) for a whole DataFrame."""
    fuzzy_df = fuzzify_dataframe(df)
    ctx = {**{col: fuzzy_df[col].values for col in fuzzy_df.columns}, **_crisp_indicators_from_df(df)}
    firing = np.stack([fn(ctx) * np.ones(len(df)) for _, fn, _ in RULES], axis=1)
    consequents = np.array([c for _, _, c in RULES])
    denom = firing.sum(axis=1)
    fri = np.where(denom > 1e-6, (firing * consequents).sum(axis=1) / np.where(denom > 1e-6, denom, 1), 0.5)
    firing_df = pd.DataFrame(firing, columns=[f"Rule{i+1}" for i in range(len(RULES))], index=df.index)
    return firing_df, fri


def compute_rule_firing_record(record: dict):
    """Returns (firing_dict, fuzzy_risk_index) for a single patient record.
    `record` must contain all of NUMERIC_COLS + CAT_COLS + 'FastingBS'."""
    fuzzy_vals = fuzzify_record(record)
    ctx = {**fuzzy_vals, **_crisp_indicators_from_record(record)}
    firings = {}
    weighted_sum, total = 0.0, 0.0
    for name, fn, consequent in RULES:
        f = float(fn(ctx))
        firings[name] = f
        weighted_sum += f * consequent
        total += f
    fri = weighted_sum / total if total > 1e-6 else 0.5
    return firings, fri, fuzzy_vals
