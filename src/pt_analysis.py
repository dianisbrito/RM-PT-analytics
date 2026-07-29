"""
Proficiency testing performance evaluation, following ISO 13528 / ISO 17043:
robust assigned value (median-based, Algorithm A style), a
fitness-for-purpose standard deviation for proficiency assessment (SDPA),
and z-scores / z'-scores classifying each participant's performance.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def mad_e(x: np.ndarray) -> float:
    """Normalized median absolute deviation (MADe), a robust SD estimator."""
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def algorithm_a(x: np.ndarray, max_iter: int = 50, tol: float = 1e-6):
    """ISO 13528 Algorithm A: iterative robust mean and SD via Huber's
    k=1.5 winsorization."""
    x = np.asarray(x, dtype=float)
    x_star = np.median(x)
    s_star = mad_e(x)
    if s_star == 0:
        s_star = np.std(x, ddof=1) or 1e-6

    for _ in range(max_iter):
        delta = 1.5 * s_star
        x_winsorized = np.clip(x, x_star - delta, x_star + delta)
        new_x_star = x_winsorized.mean()
        new_s_star = 1.134 * np.sqrt(((x_winsorized - new_x_star) ** 2).sum() / (len(x) - 1))
        if abs(new_x_star - x_star) < tol and abs(new_s_star - s_star) < tol:
            x_star, s_star = new_x_star, new_s_star
            break
        x_star, s_star = new_x_star, new_s_star

    return x_star, s_star


@dataclass
class PTResult:
    assigned_value: float
    sdpa: float
    u_assigned: float
    scored: pd.DataFrame


def evaluate_pt_round(df: pd.DataFrame, value_col: str = "reported_similarity",
                       participant_col: str = "participant") -> PTResult:
    values = df[value_col].values

    assigned_value, sdpa = algorithm_a(values)
    # Standard uncertainty of the (robust, consensus-based) assigned value
    u_assigned = 1.25 * sdpa / np.sqrt(len(values))

    scored = df.copy()
    scored["z_score"] = (scored[value_col] - assigned_value) / sdpa
    scored["z_prime_score"] = (scored[value_col] - assigned_value) / np.sqrt(sdpa ** 2 + u_assigned ** 2)

    def classify(z):
        az = abs(z)
        if az <= 2:
            return "Satisfactory"
        elif az < 3:
            return "Questionable"
        else:
            return "Unsatisfactory"

    scored["performance"] = scored["z_score"].apply(classify)
    scored = scored.sort_values("z_score", ascending=False).reset_index(drop=True)

    return PTResult(assigned_value=assigned_value, sdpa=sdpa, u_assigned=u_assigned, scored=scored)
