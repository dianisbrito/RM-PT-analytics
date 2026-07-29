"""
Stability assessment (short-term and long-term) following the classical
ISO Guide 35 linear regression approach: fit value vs. time, test whether
the slope is statistically significant, and compare the potential drift
over the intended period against the target uncertainty.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StabilityResult:
    slope: float
    intercept: float
    se_slope: float
    t_statistic: float
    p_value: float
    slope_significant: bool
    projected_drift: float  # |slope| * max_time
    max_time: float


def assess_stability(df: pd.DataFrame, value_col: str, time_col: str, alpha: float = 0.05) -> StabilityResult:
    x = df[time_col].values.astype(float)
    y = df[value_col].values.astype(float)

    reg = stats.linregress(x, y)
    slope, intercept = reg.slope, reg.intercept
    se_slope = reg.stderr

    n = len(x)
    t_stat = slope / se_slope if se_slope > 0 else 0
    p_value = reg.pvalue
    slope_significant = p_value < alpha

    max_time = x.max()
    projected_drift = abs(slope) * max_time

    return StabilityResult(
        slope=slope, intercept=intercept, se_slope=se_slope, t_statistic=t_stat,
        p_value=p_value, slope_significant=slope_significant,
        projected_drift=projected_drift, max_time=max_time,
    )


def stability_verdict(result: StabilityResult, sigma_pt: float, c: float = 0.3) -> dict:
    """Fitness-for-purpose criterion: projected drift over the study
    period should not exceed c * sigma_pt (classical c = 0.3)."""
    criterion = c * sigma_pt
    passes = result.projected_drift <= criterion
    return {
        "criterion_value": criterion,
        "projected_drift": result.projected_drift,
        "passes": passes,
        "message": (
            f"Projected drift ({result.projected_drift:.4f}) {'≤' if passes else '>'} "
            f"{c} × σ_pt ({criterion:.4f}) → material is "
            f"{'sufficiently stable' if passes else 'NOT sufficiently stable'} over the studied period"
        ),
    }
