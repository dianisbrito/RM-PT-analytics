"""
Homogeneity assessment for reference material production, following the
classical ANOVA-based approach used under ISO Guide 35 / ISO 17034:
between-unit variability is estimated and compared against a
fitness-for-purpose target uncertainty contribution.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class HomogeneityResult:
    ms_within: float
    ms_between: float
    f_statistic: float
    p_value: float
    s_within: float
    s_between_estimate: float
    ubb: float  # between-unit standard uncertainty contribution
    n_units: int
    n_replicates: int
    grand_mean: float


def assess_homogeneity(df: pd.DataFrame, value_col: str, unit_col: str = "unit") -> HomogeneityResult:
    groups = [g[value_col].values for _, g in df.groupby(unit_col)]
    n_units = len(groups)
    n_replicates = len(groups[0])

    f_stat, p_value = stats.f_oneway(*groups)

    grand_mean = df[value_col].mean()
    group_means = df.groupby(unit_col)[value_col].mean()

    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_within = n_units * (n_replicates - 1)
    ms_within = ss_within / df_within

    ss_between = n_replicates * ((group_means - grand_mean) ** 2).sum()
    df_between = n_units - 1
    ms_between = ss_between / df_between

    s_within = np.sqrt(ms_within)

    # Between-unit standard uncertainty contribution (u_bb), classical
    # ISO Guide 35 estimator; floored at 0 if MS_between < MS_within.
    variance_between_est = (ms_between - ms_within) / n_replicates
    ubb = np.sqrt(max(variance_between_est, 0))

    return HomogeneityResult(
        ms_within=ms_within, ms_between=ms_between, f_statistic=f_stat, p_value=p_value,
        s_within=s_within, s_between_estimate=np.sqrt(max(variance_between_est, 0)),
        ubb=ubb, n_units=n_units, n_replicates=n_replicates, grand_mean=grand_mean,
    )


def homogeneity_verdict(result: HomogeneityResult, sigma_pt: float, c: float = 0.3) -> dict:
    """Fitness-for-purpose criterion: u_bb should not exceed c * sigma_pt
    (classical c = 0.3, per ISO 13528 / IUPAC harmonized protocol)."""
    criterion = c * sigma_pt
    passes = result.ubb <= criterion
    return {
        "criterion_value": criterion,
        "ubb": result.ubb,
        "passes": passes,
        "message": (
            f"u_bb ({result.ubb:.4f}) {'≤' if passes else '>'} {c} × σ_pt ({criterion:.4f}) "
            f"→ material is {'sufficiently homogeneous' if passes else 'NOT sufficiently homogeneous'}"
        ),
    }
