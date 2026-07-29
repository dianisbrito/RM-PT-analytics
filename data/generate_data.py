"""
Synthetic data generator for the Reference Material / Proficiency Testing
analytics demo.

Generates fully synthetic data across the four stages of the reference
material lifecycle demonstrated in this repo:

1. Value assignment (Gower similarity to a reference taxon)
2. Homogeneity assessment (between-unit variability, ANOVA-based)
3. Stability assessment (short-term and long-term trend over time)
4. Proficiency testing performance comparison (participant z-scores)

Inspired by the general workflow of qualitative reference material
production for insect taxonomic identification (ISO 17034) and proficiency
testing schemes (ISO 17043), but with entirely fictional taxa, institutions,
and numeric values. No real specimen, institutional, or personnel data is
used anywhere in this repository.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

# ------------------------------------------------------------------
# Fictional taxa (generic scale-insect-style names, not real species
# records tied to any institution)
# ------------------------------------------------------------------
TAXA = {
    "Rf": "Referenceus typicus (Reference taxon)",
    "Sa": "Similis alpha",
    "Sb": "Similis beta",
    "Dc": "Distinctus clarus",
}

CHARACTER_NAMES = [f"char_{i:02d}" for i in range(1, 21)]
QUALITATIVE_FLAGS = np.array([True] * 8 + [False] * 12)  # 8 qualitative, 12 quantitative


# ====================================================================
# 1) VALUE ASSIGNMENT DATA (Gower)
# ====================================================================
def generate_gower_specimens(n_per_taxon=25):
    """Simulate morphological character measurements for specimens of
    4 taxa, with the reference taxon 'Rf' as the identity standard."""
    rows = []
    # Each taxon has a "true" underlying profile per character (0-1 scale
    # for qualitative similarity-coded characters, arbitrary units for
    # quantitative ones); specimens are noisy draws around that profile.
    taxon_profiles = {}
    base_profile = RNG.uniform(0.3, 0.9, len(CHARACTER_NAMES))
    for taxon in TAXA:
        if taxon == "Rf":
            taxon_profiles[taxon] = base_profile
        else:
            # Other taxa diverge on a random subset of characters
            profile = base_profile.copy()
            n_diverge = RNG.integers(4, 10)
            diverge_idx = RNG.choice(len(CHARACTER_NAMES), n_diverge, replace=False)
            profile[diverge_idx] = RNG.uniform(0, 1, n_diverge)
            taxon_profiles[taxon] = profile

    for taxon in TAXA:
        for i in range(1, n_per_taxon + 1):
            specimen_id = f"{taxon}-U-{i:03d}"
            values = taxon_profiles[taxon] + RNG.normal(0, 0.06, len(CHARACTER_NAMES))
            values = np.clip(values, 0, 1)
            row = {"specimen_id": specimen_id, "taxon": taxon}
            row.update({name: val for name, val in zip(CHARACTER_NAMES, values)})
            rows.append(row)

    specimens = pd.DataFrame(rows)

    metadata = pd.DataFrame({
        "character": CHARACTER_NAMES,
        "descripcion": [f"Morphological character {i+1} (illustrative)" for i in range(len(CHARACTER_NAMES))],
        "cualitativa": np.where(QUALITATIVE_FLAGS, "Si", "No"),
        "num_estados": np.where(QUALITATIVE_FLAGS, RNG.integers(2, 5, len(CHARACTER_NAMES)), np.nan),
        "valor_referencia": taxon_profiles["Rf"],
    })

    return specimens, metadata


# ====================================================================
# 2) HOMOGENEITY DATA
# ====================================================================
def generate_homogeneity_data(n_units=10, n_replicates=2, between_unit_sd=0.015, within_unit_sd=0.02):
    """Simulate a homogeneity study: n_units production units, each
    measured in duplicate, on the Gower similarity score to the
    reference taxon (the 'measurand' carried through this material)."""
    true_mean = 0.92  # target similarity for a correctly-identified, well-prepared unit
    unit_effects = RNG.normal(0, between_unit_sd, n_units)

    rows = []
    for u in range(1, n_units + 1):
        unit_mean = true_mean + unit_effects[u - 1]
        for r in range(1, n_replicates + 1):
            value = unit_mean + RNG.normal(0, within_unit_sd)
            rows.append({"unit": f"U{u:02d}", "replicate": r, "gower_similarity": np.clip(value, 0, 1)})

    return pd.DataFrame(rows)


# ====================================================================
# 3) STABILITY DATA
# ====================================================================
def generate_stability_data():
    """Simulate short-term (accelerated, e.g. elevated temperature) and
    long-term (real-time storage) stability studies, tracking the same
    Gower similarity measurand over time."""
    true_mean = 0.92

    # Short-term: measured at 0, 1, 2, 4, 8 weeks under stress conditions
    short_term_weeks = [0, 1, 2, 4, 8]
    short_term_slope = -0.0015  # small, statistically non-significant drift
    short_rows = []
    for week in short_term_weeks:
        for rep in range(1, 4):
            value = true_mean + short_term_slope * week + RNG.normal(0, 0.015)
            short_rows.append({"time_weeks": week, "replicate": rep, "gower_similarity": np.clip(value, 0, 1)})
    short_term = pd.DataFrame(short_rows)

    # Long-term: measured at 0, 3, 6, 12, 18, 24 months under storage conditions
    long_term_months = [0, 3, 6, 12, 18, 24]
    long_term_slope = -0.0008
    long_rows = []
    for month in long_term_months:
        for rep in range(1, 4):
            value = true_mean + long_term_slope * month + RNG.normal(0, 0.012)
            long_rows.append({"time_months": month, "replicate": rep, "gower_similarity": np.clip(value, 0, 1)})
    long_term = pd.DataFrame(long_rows)

    return short_term, long_term


# ====================================================================
# 4) PROFICIENCY TESTING DATA
# ====================================================================
def generate_pt_data(n_participants=18):
    """Simulate a proficiency-testing round where participant
    laboratories report their computed Gower similarity of a test
    specimen to the reference taxon. Most cluster near the true value;
    a few are deliberately simulated as outliers/discrepant."""
    true_value = 0.92
    participants = [f"LAB-{i:02d}" for i in range(1, n_participants + 1)]

    values = RNG.normal(true_value, 0.02, n_participants)
    # Inject a few realistic discrepant participants
    outlier_idx = RNG.choice(n_participants, 3, replace=False)
    values[outlier_idx[0]] += 0.10
    values[outlier_idx[1]] -= 0.12
    values[outlier_idx[2]] += 0.06

    values = np.clip(values, 0, 1)

    df = pd.DataFrame({"participant": participants, "reported_similarity": values})
    return df


def save_all(output_dir="data"):
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    specimens, metadata = generate_gower_specimens()
    specimens.to_csv(out / "gower_specimens.csv", index=False)
    metadata.to_csv(out / "gower_metadata.csv", index=False)

    homogeneity = generate_homogeneity_data()
    homogeneity.to_csv(out / "homogeneity.csv", index=False)

    short_term, long_term = generate_stability_data()
    short_term.to_csv(out / "stability_short_term.csv", index=False)
    long_term.to_csv(out / "stability_long_term.csv", index=False)

    pt_data = generate_pt_data()
    pt_data.to_csv(out / "pt_results.csv", index=False)

    return {
        "specimens": specimens, "metadata": metadata, "homogeneity": homogeneity,
        "short_term": short_term, "long_term": long_term, "pt_data": pt_data,
    }


if __name__ == "__main__":
    result = save_all()
    print("Generated all synthetic datasets:")
    for k, v in result.items():
        print(f"  {k}: {v.shape}")
