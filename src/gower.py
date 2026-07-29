"""
Gower similarity coefficient and Minimum Dimension Analysis (MDA) weighting
for taxonomic/morphological value assignment.

Implements the same core methodology as the ISO/TR 79:2015-style approach:
per-character similarity, weighted (MDA-derived weights) and unweighted
overall Gower similarity, and a between/within-taxon variance ratio
(lambda^2_Ni) used to identify the most discriminating characters.
"""

import numpy as np
import pandas as pd


def compute_character_similarity(specimens: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Per-character similarity g_i = 1 - |x_i - x0| / r_i for every specimen."""
    character_cols = metadata["character"].tolist()
    sims = pd.DataFrame(index=specimens.index, columns=character_cols, dtype=float)

    for _, meta_row in metadata.iterrows():
        col = meta_row["character"]
        x0 = meta_row["valor_referencia"]

        if meta_row["cualitativa"] == "Si" and not pd.isna(meta_row["num_estados"]):
            r = max(meta_row["num_estados"] - 1, 1)
        else:
            r = specimens[col].max() - specimens[col].min()
            if r == 0 or pd.isna(r):
                r = 1

        g = 1 - (specimens[col] - x0).abs() / r
        sims[col] = g.clip(0, 1)

    return sims


def compute_mda_weights(similarities: pd.DataFrame, taxa: pd.Series) -> pd.DataFrame:
    """Minimum Dimension Analysis: lambda^2_Ni = sigma_b^2 / (sigma_b^2 + sigma_w^2)
    per character, used as a discriminating-power weight."""
    results = []
    for col in similarities.columns:
        values = similarities[col]
        valid = values.notna()
        if valid.sum() < 5:
            results.append({"character": col, "sigma_b2": 0, "sigma_w2": 0, "lambda2_Ni": 0})
            continue

        v = values[valid]
        g = taxa[valid]
        overall_mean = v.mean()
        group_means = v.groupby(g).mean()
        group_counts = v.groupby(g).count()

        sigma_b2 = float((group_counts * (group_means - overall_mean) ** 2).sum() / group_counts.sum())
        within_vars = v.groupby(g).var(ddof=1).fillna(0)
        sigma_w2 = float(within_vars.mean())
        if sigma_w2 == 0:
            sigma_w2 = 1e-4

        lambda2 = sigma_b2 / (sigma_b2 + sigma_w2)
        results.append({"character": col, "sigma_b2": sigma_b2, "sigma_w2": sigma_w2, "lambda2_Ni": lambda2})

    df = pd.DataFrame(results)
    total = df["lambda2_Ni"].sum()
    df["peso"] = df["lambda2_Ni"] / total if total > 0 else 1 / len(df)
    return df.sort_values("lambda2_Ni", ascending=False).reset_index(drop=True)


def gower_similarity(similarities: pd.DataFrame, weights: pd.Series | None = None) -> pd.Series:
    """Overall Gower similarity per specimen — weighted if `weights`
    (indexed by character) is provided, unweighted (simple mean) otherwise."""
    if weights is None:
        return similarities.mean(axis=1)
    w = weights.reindex(similarities.columns).fillna(0)
    return (similarities * w).sum(axis=1) / w.sum()
