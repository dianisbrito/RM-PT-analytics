"""
Reference Material & Proficiency Testing Analytics — Streamlit demo

Demonstrates the statistical workflow behind producing a qualitative
reference material for taxonomic identification (ISO 17034) and running
a proficiency testing scheme around it (ISO 17043), using the Gower
similarity coefficient as the measurand carried through all four stages:
value assignment, homogeneity, stability, and participant performance
comparison.

Note: all data is synthetic — see data/generate_data.py.
Run with: streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "data"))

from generate_data import save_all
from gower import compute_character_similarity, compute_mda_weights, gower_similarity
from homogeneity import assess_homogeneity, homogeneity_verdict
from stability import assess_stability, stability_verdict
from pt_analysis import evaluate_pt_round

st.set_page_config(page_title="RM & Proficiency Testing Analytics", layout="wide")

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_data():
    paths = [DATA_DIR / f for f in [
        "gower_specimens.csv", "gower_metadata.csv", "homogeneity.csv",
        "stability_short_term.csv", "stability_long_term.csv", "pt_results.csv",
    ]]
    if all(p.exists() for p in paths):
        return {
            "specimens": pd.read_csv(paths[0]),
            "metadata": pd.read_csv(paths[1]),
            "homogeneity": pd.read_csv(paths[2]),
            "short_term": pd.read_csv(paths[3]),
            "long_term": pd.read_csv(paths[4]),
            "pt_data": pd.read_csv(paths[5]),
        }
    return save_all(output_dir=str(DATA_DIR))


data = load_data()

st.title("🔬 Reference Material & Proficiency Testing Analytics")
st.caption(
    "Demo dashboard — synthetic data, inspired by the statistical workflow behind "
    "qualitative reference material production (ISO 17034) and proficiency testing "
    "(ISO 17043) for taxonomic/morphological identification. All taxa, institutions, "
    "and values are fictional — see the README for details."
)

tab_value, tab_hom, tab_stab, tab_pt = st.tabs(
    ["🧬 Value Assignment (Gower)", "⚖️ Homogeneity", "📆 Stability", "🏆 Proficiency Testing"]
)

# ====================================================================
# TAB 1 — VALUE ASSIGNMENT (GOWER)
# ====================================================================
with tab_value:
    st.subheader("Value assignment via the Gower similarity coefficient")
    st.caption(
        "Each specimen's morphological characters are compared against the reference "
        "taxon's characteristic profile. Similarity per character: "
        "g_i = 1 − |xᵢ − x₀| / rᵢ. Overall similarity can be unweighted (simple mean) "
        "or weighted using discriminating power (MDA λ²ₙᵢ) per character."
    )

    specimens = data["specimens"]
    metadata = data["metadata"]

    similarities = compute_character_similarity(specimens, metadata)
    mda = compute_mda_weights(similarities, specimens["taxon"])
    weights = mda.set_index("character")["peso"]

    specimens = specimens.copy()
    specimens["gower_weighted"] = gower_similarity(similarities, weights)
    specimens["gower_unweighted"] = gower_similarity(similarities)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Overall Gower similarity to reference taxon, by taxon**")
        method = st.radio("Method", ["Weighted (MDA)", "Unweighted"], horizontal=True)
        value_col = "gower_weighted" if method == "Weighted (MDA)" else "gower_unweighted"

        fig = px.box(specimens, x="taxon", y=value_col, color="taxon", points="all",
                     labels={value_col: "Gower similarity", "taxon": "Taxon"})
        fig.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top discriminating characters (MDA)**")
        st.caption("λ²ₙᵢ = σ²between / (σ²between + σ²within) — higher = more discriminating")
        top_mda = mda.head(10)[["character", "lambda2_Ni", "peso"]]
        st.dataframe(top_mda.style.format({"lambda2_Ni": "{:.3f}", "peso": "{:.3f}"}),
                     hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Hierarchical clustering (dendrogram) on Gower distance")
    n_per_taxon = st.slider("Specimens per taxon to display", 3, 15, 8)
    sample = (
        specimens.groupby("taxon", group_keys=False)
        .apply(lambda g: g.sample(min(n_per_taxon, len(g)), random_state=1))
        .reset_index(drop=True)
    )

    sample_sims = compute_character_similarity(sample, metadata)
    dist = 1 - sample_sims.values
    labels = sample["specimen_id"].tolist()

    fig_dendro = ff.create_dendrogram(dist, labels=labels, color_threshold=0.4)
    fig_dendro.update_layout(height=550, xaxis_title="Specimen", yaxis_title="Gower distance")
    st.plotly_chart(fig_dendro, use_container_width=True)
    st.caption(
        "Specimens clustering with the reference taxon (Rf) at low distance support a "
        "consistent identity assignment; taxa with higher within-cluster distance to Rf "
        "indicate greater morphological divergence."
    )

# ====================================================================
# TAB 2 — HOMOGENEITY
# ====================================================================
with tab_hom:
    st.subheader("Between-unit homogeneity assessment")
    st.caption(
        "Following the classical ANOVA-based approach (ISO Guide 35 / ISO 17034): "
        "the Gower similarity score is measured in duplicate across production units. "
        "Between-unit variability (u_bb) is compared against a fitness-for-purpose target."
    )

    sigma_pt_hom = st.slider("Target σ_pt for proficiency assessment (fitness-for-purpose)",
                              0.01, 0.10, 0.05, 0.005, key="sigma_hom")

    hom_df = data["homogeneity"]
    hom_result = assess_homogeneity(hom_df, "gower_similarity", "unit")
    hom_verdict = homogeneity_verdict(hom_result, sigma_pt_hom)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grand mean", f"{hom_result.grand_mean:.4f}")
    c2.metric("F statistic (ANOVA)", f"{hom_result.f_statistic:.3f}")
    c3.metric("p-value", f"{hom_result.p_value:.3f}")
    c4.metric("u_bb (between-unit)", f"{hom_result.ubb:.4f}")

    if hom_verdict["passes"]:
        st.success(hom_verdict["message"])
    else:
        st.warning(hom_verdict["message"])

    fig_hom = px.strip(hom_df, x="unit", y="gower_similarity", color="unit",
                       labels={"gower_similarity": "Gower similarity", "unit": "Production unit"})
    fig_hom.add_hline(y=hom_result.grand_mean, line_dash="dash", line_color="gray",
                      annotation_text="Grand mean")
    fig_hom.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig_hom, use_container_width=True)

    with st.expander("ANOVA detail"):
        anova_table = pd.DataFrame({
            "Source": ["Between units", "Within units"],
            "MS": [hom_result.ms_between, hom_result.ms_within],
        })
        st.dataframe(anova_table.style.format({"MS": "{:.6f}"}), hide_index=True, use_container_width=True)
        st.caption(
            f"n_units = {hom_result.n_units}, replicates per unit = {hom_result.n_replicates}. "
            f"u_bb = √max(0, (MS_between − MS_within) / replicates)"
        )

# ====================================================================
# TAB 3 — STABILITY
# ====================================================================
with tab_stab:
    st.subheader("Stability assessment")
    st.caption(
        "Classical ISO Guide 35 regression approach: fit Gower similarity vs. time, "
        "test whether the slope differs significantly from zero, and compare the "
        "projected drift over the study period against the fitness-for-purpose target."
    )

    sigma_pt_stab = st.slider("Target σ_pt for proficiency assessment (fitness-for-purpose)",
                               0.01, 0.10, 0.05, 0.005, key="sigma_stab")

    stab_type = st.radio("Study type", ["Short-term (accelerated)", "Long-term (real-time storage)"], horizontal=True)

    if stab_type == "Short-term (accelerated)":
        df_stab = data["short_term"]
        time_col = "time_weeks"
        time_label = "Time (weeks)"
    else:
        df_stab = data["long_term"]
        time_col = "time_months"
        time_label = "Time (months)"

    stab_result = assess_stability(df_stab, "gower_similarity", time_col)
    stab_verdict = stability_verdict(stab_result, sigma_pt_stab)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Slope", f"{stab_result.slope:.5f} /unit time")
    c2.metric("p-value (slope ≠ 0)", f"{stab_result.p_value:.3f}")
    c3.metric("Significant drift?", "Yes" if stab_result.slope_significant else "No")
    c4.metric("Projected drift", f"{stab_result.projected_drift:.4f}")

    if stab_verdict["passes"]:
        st.success(stab_verdict["message"])
    else:
        st.warning(stab_verdict["message"])

    fig_stab = px.scatter(df_stab, x=time_col, y="gower_similarity",
                          labels={"gower_similarity": "Gower similarity", time_col: time_label})
    x_range = np.linspace(df_stab[time_col].min(), df_stab[time_col].max(), 50)
    y_fit = stab_result.intercept + stab_result.slope * x_range
    fig_stab.add_trace(go.Scatter(x=x_range, y=y_fit, mode="lines", name="Linear fit", line=dict(color="firebrick")))
    fig_stab.update_layout(height=450)
    st.plotly_chart(fig_stab, use_container_width=True)

# ====================================================================
# TAB 4 — PROFICIENCY TESTING
# ====================================================================
with tab_pt:
    st.subheader("Proficiency testing round — participant performance")
    st.caption(
        "Following ISO 13528 / ISO 17043: the assigned value and fitness-for-purpose "
        "standard deviation (SDPA) are computed robustly from participant results using "
        "Algorithm A (iterative Huber-type robust mean/SD). Each participant is then "
        "scored with a z-score."
    )

    pt_result = evaluate_pt_round(data["pt_data"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Assigned value (robust)", f"{pt_result.assigned_value:.4f}")
    c2.metric("SDPA (robust SD)", f"{pt_result.sdpa:.4f}")
    c3.metric("u(assigned value)", f"{pt_result.u_assigned:.4f}")

    scored = pt_result.scored
    n_sat = (scored["performance"] == "Satisfactory").sum()
    n_q = (scored["performance"] == "Questionable").sum()
    n_unsat = (scored["performance"] == "Unsatisfactory").sum()

    c4, c5, c6 = st.columns(3)
    c4.metric("✅ Satisfactory (|z|≤2)", n_sat)
    c5.metric("⚠️ Questionable (2<|z|<3)", n_q)
    c6.metric("❌ Unsatisfactory (|z|≥3)", n_unsat)

    color_map = {"Satisfactory": "#2E7D32", "Questionable": "#FF9800", "Unsatisfactory": "#D32F2F"}
    fig_z = px.bar(
        scored, x="participant", y="z_score", color="performance",
        color_discrete_map=color_map,
        labels={"z_score": "z-score", "participant": "Participant lab"},
    )
    fig_z.add_hline(y=2, line_dash="dash", line_color="orange")
    fig_z.add_hline(y=-2, line_dash="dash", line_color="orange")
    fig_z.add_hline(y=3, line_dash="dot", line_color="red")
    fig_z.add_hline(y=-3, line_dash="dot", line_color="red")
    fig_z.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig_z, use_container_width=True)

    st.dataframe(
        scored[["participant", "reported_similarity", "z_score", "z_prime_score", "performance"]]
        .style.format({"reported_similarity": "{:.4f}", "z_score": "{:.2f}", "z_prime_score": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )



