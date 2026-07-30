# Reference Material & Proficiency Testing Analytics

**Author:** Diana Brito Hoyos — Biologist & Biostatistician | Data Analyst

🔗 **[Live demo](https://rm-pt-analytics.streamlit.app/)**

An interactive dashboard demonstrating the statistical workflow behind producing a **qualitative reference material for taxonomic/morphological identification** (under **ISO 17034**) and running a **proficiency testing scheme** around it (under **ISO 17043**).

> ⚠️ **Note on data and origin:** This project is inspired by real applied work in reference-material production and value assignment for insect taxonomic identification using the Gower similarity coefficient — part of my role as a data analysis professional in a phytosanitary diagnostic laboratory. **All taxa, specimen data, institutional names, and numeric values in this repository are entirely synthetic** (see [`data/generate_data.py`](./data/generate_data.py)). No real specimen records, personnel names, or institutional/production documents are used. The methodology (Gower similarity, MDA weighting, ANOVA-based homogeneity, Guide-35-style stability regression, and ISO 13528 Algorithm A for proficiency testing) reflects standard, published statistical practice in this field.

---

## The unifying idea

All four stages of the reference-material lifecycle shown here are tied together by the **same measurand**: the Gower similarity of a specimen to its reference taxon. This mirrors how a qualitative identification reference material is actually characterized and monitored in practice — a single, well-defined statistic carried from value assignment all the way through to how participant laboratories are scored in a proficiency test.

## Dashboard tabs

1. **🧬 Value Assignment (Gower)** — Per-character similarity (`gᵢ = 1 − |xᵢ − x₀| / rᵢ`), weighted (Minimum Dimension Analysis, λ²ₙᵢ) vs. unweighted overall similarity, and a hierarchical clustering dendrogram on Gower distance.
2. **⚖️ Homogeneity** — Between/within-unit ANOVA on duplicate measurements across production units, with the classical `u_bb ≤ 0.3·σ_pt` fitness-for-purpose criterion (ISO Guide 35 / ISO 17034).
3. **📆 Stability** — Short-term (accelerated) and long-term (real-time storage) linear regression of the measurand vs. time, testing slope significance and projected drift against the target uncertainty.
4. **🏆 Proficiency Testing** — ISO 13528 Algorithm A robust assigned value and SDPA, participant z-scores and z′-scores, and a satisfactory/questionable/unsatisfactory performance breakdown.

## Project structure

```
rm-pt-analytics/
├── README.md
├── requirements.txt
├── app.py                       ← Streamlit dashboard (4 tabs)
├── src/
│   ├── gower.py                 ← Gower similarity + MDA weighting
│   ├── homogeneity.py           ← ANOVA-based homogeneity assessment
│   ├── stability.py             ← regression-based stability assessment
│   └── pt_analysis.py           ← ISO 13528 Algorithm A + z-scores
└── data/
    └── generate_data.py         ← synthetic data generator (all 4 stages)
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
(the app generates all synthetic datasets automatically on first run)

## Tech stack

`Python` · `scipy (ANOVA, regression)` · `Streamlit` · `pandas` · `plotly` · statistical methods aligned with `ISO 17034` · `ISO 17043` · `ISO 13528` · `ISO Guide 35`
