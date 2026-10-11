"""Global and per-session explanations for the LightGBM model (SHAP).

SHAP values describe how the *model* uses each feature for a prediction. They
are associations learned from data, not causal effects on the athlete.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES, PRETTY


def shap_values(model, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    import shap
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X[FEATURES])
    base = float(np.ravel(explainer.expected_value)[0])
    return pd.DataFrame(vals, columns=FEATURES, index=X.index), base


def global_importance(shap_df: pd.DataFrame) -> pd.DataFrame:
    imp = shap_df.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index, "label": [PRETTY.get(f, f) for f in imp.index],
                         "mean_abs_shap_bpm": imp.to_numpy()})


def local_explanation(shap_row: pd.Series, feature_row: pd.Series, base: float, top: int = 8) -> pd.DataFrame:
    """Top contributions for one prediction, in bpm, largest magnitude first."""
    contrib = shap_row[FEATURES]
    order = contrib.abs().sort_values(ascending=False).index[:top]
    return pd.DataFrame({
        "feature": order,
        "label": [PRETTY.get(f, f) for f in order],
        "value": [feature_row[f] for f in order],
        "contribution_bpm": contrib[order].to_numpy(),
    }).assign(base_bpm=base)
