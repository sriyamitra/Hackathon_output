import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nonprofit Resilience Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    /* KPI cards */
    .metric-card {
        background: #fdf6e3; border-radius: 8px;
        padding: 16px 20px; text-align: center;
        border: 1px solid #e8dcc8;
    }
    .metric-label { font-size: 13px; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a1a2e; }
    .metric-sub   { font-size: 12px; color: #6c757d; margin-top: 2px; }

    /* Section titles */
    h1 { font-size: 1.8rem !important; }

    /* Metric definition boxes */
    .metric-def {
        background: #fdf6e3; border-left: 4px solid #d4a846;
        border-radius: 0 6px 6px 0; padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-def-title { font-weight: 700; font-size: 13px; color: #1a1a2e; }
    .metric-def-body  { font-size: 12px; color: #555; margin-top: 3px; }
    .metric-def-aim   { font-size: 11px; color: #4CAF50; margin-top: 4px; font-style: italic; }
</style>
""", unsafe_allow_html=True)

TIER_COLORS = {
    "Thriving": "#4CAF50",
    "Stable":   "#FFC107",
    "At-Risk":  "#FF9800",
    "Fragile":  "#F44336",
    "Unknown":  "#9E9E9E",
}
TIER_ORDER = ["Fragile", "At-Risk", "Stable", "Thriving"]

BENCHMARK_TIER_COLORS = {
    "Thriving": "#4CAF50",
    "Stable":   "#8BC34A",
    "Moderate": "#FFC107",
    "At Risk":  "#FF9800",
    "Critical": "#F44336",
    "Unknown":  "#9E9E9E",
}
BENCHMARK_TIER_ORDER = ["Critical", "At Risk", "Moderate", "Stable", "Thriving"]

NTEE_LABELS = {
    "A": "Arts & Culture",    "B": "Education",         "C": "Environment",
    "D": "Animal Welfare",    "E": "Health",             "F": "Mental Health",
    "G": "Disease",           "H": "Medical Research",  "I": "Crime & Legal",
    "J": "Employment",        "K": "Food & Agriculture","L": "Housing",
    "M": "Public Safety",     "N": "Recreation",        "O": "Youth Development",
    "P": "Human Services",    "Q": "International",     "R": "Civil Rights",
    "S": "Community Dev.",    "T": "Public Benefit",    "U": "Science & Tech",
    "V": "Social Science",    "W": "Public Affairs",    "X": "Religion",
    "Y": "Mutual Benefit",    "Z": "Unknown",
}

# US Census regions
REGION_MAP = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast", "RI": "Northeast", "VT": "Northeast",
    "IL": "Midwest",   "IN": "Midwest",   "IA": "Midwest",   "KS": "Midwest",
    "MI": "Midwest",   "MN": "Midwest",   "MO": "Midwest",   "NE": "Midwest",
    "ND": "Midwest",   "OH": "Midwest",   "SD": "Midwest",   "WI": "Midwest",
    "AL": "South",     "AR": "South",     "DE": "South",     "FL": "South",
    "GA": "South",     "KY": "South",     "LA": "South",     "MD": "South",
    "MS": "South",     "NC": "South",     "OK": "South",     "SC": "South",
    "TN": "South",     "TX": "South",     "VA": "South",     "WV": "South",  "DC": "South",
    "AK": "West",      "AZ": "West",      "CA": "West",      "CO": "West",
    "HI": "West",      "ID": "West",      "MT": "West",      "NV": "West",
    "NM": "West",      "OR": "West",      "UT": "West",      "WA": "West",   "WY": "West",
}

SIZE_TIERS = [
    "Micro (<$100K)",
    "Small ($100K–$1M)",
    "Medium ($1M–$10M)",
    "Large ($10M–$100M)",
    "Major (>$100M)",
]

# ── Data loading & scoring (990-PF overview tab) ──────────────────────────────
# Pipeline mirrors 990PF_old_dataextraction.ipynb exactly:
#   all_records.parquet  →  cast types  →  merge benchmarking  →  metrics  →  score
@st.cache_data(show_spinner="Loading 990-PF data…")
def load_and_score():
    df = pd.read_parquet("all_records.parquet")

    # ── Cast types (notebook Cell 5) ──────────────────────────────────────────
    if "tax_year" in df.columns:
        df["tax_year"] = pd.to_numeric(df["tax_year"], errors="coerce")
    elif "filing_year" in df.columns:
        df["tax_year"] = pd.to_numeric(df["filing_year"], errors="coerce")

    money_cols = [
        "dividends", "cap_gains", "other_income", "total_revenue",
        "net_invest_income", "operating_expenses", "total_expenses",
        "grants_paid", "assets_boy", "assets_eoy", "assets_eoy_fmv",
        "fmv_assets_eoy", "liabilities_eoy", "net_assets_boy",
        "net_assets_eoy", "excise_tax",
    ]
    existing = [c for c in money_cols if c in df.columns]
    df[existing] = df[existing].apply(pd.to_numeric, errors="coerce").fillna(0)

    # ── Merge benchmarking for sector/location (notebook Cells 7-8) ───────────
    bench = pd.read_excel(
        "990 PF_Benchmarking_Final.xlsx",
        usecols=["ein", "bmf_city", "bmf_state", "bmf_ntee_code"],
    )
    bench["ein"] = bench["ein"].astype(str).str.strip()
    bench = bench.drop_duplicates(subset="ein", keep="last")

    df["ein"] = df["ein"].astype(str).str.strip()
    df = df.merge(bench, on="ein", how="left")

    df["ntee_broad"] = df["bmf_ntee_code"].astype(str).str[0].str.upper()
    df["ntee_broad"] = df["ntee_broad"].where(df["ntee_broad"].isin(NTEE_LABELS), "Unknown")
    df["sector"]     = df["ntee_broad"].map(NTEE_LABELS).fillna("Unknown")

    # ── Core metrics (notebook Cell 16) ───────────────────────────────────────
    dfc = df.dropna(subset=["fmv_assets_eoy"]).copy()
    dfc = dfc[dfc["fmv_assets_eoy"] > 0]

    dfc["payout_ratio"]       = dfc["grants_paid"] / dfc["fmv_assets_eoy"]
    dfc["operating_margin"]   = (
        (dfc["total_revenue"] - dfc["total_expenses"])
        / dfc["total_revenue"].replace(0, pd.NA)
    )
    dfc["months_of_reserves"] = (
        dfc["net_assets_eoy"] / (dfc["total_expenses"].replace(0, pd.NA) / 12)
    )
    dfc["admin_overhead"] = (
        dfc["operating_expenses"] / dfc["total_expenses"].replace(0, pd.NA)
    )

    # ── Aggregate to one row per EIN (notebook Cell 18) ───────────────────────
    dfc_s = dfc.sort_values(["ein", "tax_year"])
    dft = dfc_s.groupby("ein").agg(
        org_name           = ("org_name",          "last"),
        state              = ("bmf_state",         "last"),
        city               = ("bmf_city",          "last"),
        sector             = ("sector",            "last"),
        ntee_code          = ("bmf_ntee_code",     "last"),
        n_years            = ("tax_year",          "nunique"),
        fmv_assets_first   = ("fmv_assets_eoy",    "first"),
        fmv_assets_last    = ("fmv_assets_eoy",    "last"),
        payout_ratio       = ("payout_ratio",       "last"),
        operating_margin   = ("operating_margin",   "last"),
        months_of_reserves = ("months_of_reserves", "last"),
        admin_overhead     = ("admin_overhead",     "last"),
    ).reset_index()

    dft["asset_cagr"] = np.where(
        (dft["n_years"] >= 2) & (dft["fmv_assets_first"] > 0),
        (dft["fmv_assets_last"] / dft["fmv_assets_first"]) ** (1 / dft["n_years"]) - 1,
        np.nan,
    )

    # ── Resilience scoring (notebook Cell 21) ─────────────────────────────────
    def pct_rank(s):
        return s.rank(pct=True, na_option="keep") * 100

    dft["s_payout_ratio"]     = pct_rank(dft["payout_ratio"])
    dft["s_operating_margin"] = pct_rank(dft["operating_margin"])
    dft["s_months_reserves"]  = pct_rank(dft["months_of_reserves"])
    dft["s_asset_cagr"]       = pct_rank(dft["asset_cagr"])
    dft["s_admin_overhead"]   = 100 - pct_rank(dft["admin_overhead"])

    WEIGHTS = [
        ("s_payout_ratio",     0.25),
        ("s_operating_margin", 0.25),
        ("s_months_reserves",  0.20),
        ("s_asset_cagr",       0.20),
        ("s_admin_overhead",   0.10),
    ]

    def weighted_score(row):
        ts, tw = 0, 0
        for col, w in WEIGHTS:
            v = row[col]
            if pd.notna(v):
                ts += v * w
                tw += w
        return round(ts / tw, 1) if tw > 0 else np.nan

    dft["resilience_score"] = dft.apply(weighted_score, axis=1)

    def assign_tier(s):
        if pd.isna(s):  return "Unknown"
        if s >= 75:     return "Thriving"
        if s >= 50:     return "Stable"
        if s >= 25:     return "At-Risk"
        return                 "Fragile"

    dft["resilience_tier"] = dft["resilience_score"].apply(assign_tier)
    return dft


# ── Benchmarking data loading ─────────────────────────────────────────────────
# Reads pre-computed peer benchmarking results from step4 notebooks.
@st.cache_data(show_spinner="Loading benchmarking data…")
def load_benchmarking_data():
    pf  = pd.read_csv("step4_990pf_3.csv", low_memory=False)
    pub = pd.read_csv("step4_990_3.csv",   low_memory=False)
    df  = pd.concat([pf, pub], ignore_index=True)

    # ── Normalize columns to match dashboard expectations ─────────────────────
    df["state"]  = df["bmf_state"]
    df["city"]   = df["bmf_city"]
    df["is_pf"]  = df["form_type"] == "990PF"

    # peer_benchmark_score is 0–1 in CSV → scale to 0–100
    df["benchmark_score"] = (df["peer_benchmark_score"] * 100).round(1)

    # Strip prefix from performance_tier: "1_Thriving" → "Thriving"
    tier_map = {
        "1_Thriving": "Thriving",
        "2_Stable":   "Stable",
        "3_Moderate": "Moderate",
        "4_At Risk":  "At Risk",
        "5_Critical": "Critical",
    }
    df["benchmark_tier"] = df["performance_tier"].map(tier_map).fillna("Unknown")

    # Map size_bucket to display labels
    size_map = {
        "1_Micro":   "Micro (<$100K)",
        "2_Small":   "Small ($100K–$1M)",
        "3_Medium":  "Medium ($1M–$10M)",
        "4_Large":   "Large ($10M–$100M)",
        "5_Major":   "Major (>$100M)",
    }
    df["size_tier"] = df["size_bucket"].map(size_map).fillna("Unknown")

    # Align months column name (990 CSV uses months_of_reserves_latest)
    if "months_of_reserves_latest" in df.columns:
        df["months_reserves_latest"] = df["months_of_reserves_latest"]

    return df


# ── Survival rate data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing survival rates…")
def compute_pf_survival():
    # all_records.parquet is the full 990-PF multi-year panel (160K+ filings).
    # nonprofit_master.parquet only covers a partial year range, so we use all_records here.
    raw = pd.read_parquet("all_records.parquet")
    raw["tax_year"] = pd.to_numeric(raw["tax_year"], errors="coerce")
    raw["ein"]      = raw["ein"].astype(str).str.strip()

    years_sorted = sorted(raw["tax_year"].dropna().unique().astype(int))
    ein_by_year  = {yr: set(raw[raw["tax_year"] == yr]["ein"].dropna())
                    for yr in years_sorted}
    filer_counts = {yr: len(eins) for yr, eins in ein_by_year.items()}
    peak         = max(filer_counts.values())

    # Exclude years with < 30 % of peak filers (IRS data-publication lag)
    incomplete   = {yr for yr, n in filer_counts.items() if n < peak * 0.30}
    complete     = sorted(yr for yr in years_sorted if yr not in incomplete)

    rows = []
    for yr_a, yr_b in zip(complete[:-1], complete[1:]):
        if yr_b - yr_a != 1:
            continue
        a, b = ein_by_year[yr_a], ein_by_year[yr_b]
        rows.append({
            "transition":    f"{yr_a}→{yr_b}",
            "year_from":     yr_a,
            "total_from":    len(a),
            "total_to":      len(b),
            "survivors":     len(a & b),
            "dropouts":      len(a - b),
            "new_entrants":  len(b - a),
            "survival_rate": round(len(a & b) / len(a) * 100, 1) if a else 0,
        })
    return pd.DataFrame(rows), sorted(incomplete)


# ── Unique PF foundation count (from full panel) ──────────────────────────────
@st.cache_data(show_spinner=False)
def count_pf_foundations():
    """Count distinct EINs in the full 990-PF panel (all_records.parquet)."""
    raw = pd.read_parquet("all_records.parquet", columns=["ein"])
    return int(raw["ein"].astype(str).str.strip().nunique())


# ── 990 Public Charity data loading ──────────────────────────────────────────
@st.cache_data(show_spinner="Loading 990 public charity data…")
def load_990_data():
    dr = pd.read_parquet("nonprofit_990_resilience_scores.parquet")

    # Strip emoji prefix from tier labels ("🟡 Stable" → "Stable")
    dr["resilience_tier"] = (
        dr["resilience_tier"]
        .str.replace(r"^[^\w]+", "", regex=True)
        .str.strip()
    )

    # The 'state' column is a duplicate of 'ein' — drop before merging
    if "state" in dr.columns:
        dr = dr.drop(columns=["state"])

    # Merge geographic/sector info from step4 990 CSV
    dc = pd.read_csv("step4_990_3.csv", low_memory=False)
    dc["ein"] = dc["ein"].astype(str).str.strip()
    dr["ein"] = dr["ein"].astype(str).str.strip()

    dc_geo = (
        dc[["ein", "bmf_state", "bmf_city", "sector", "bmf_ntee_code"]]
        .drop_duplicates("ein")
    )
    dr = dr.merge(dc_geo, on="ein", how="left")
    dr.rename(columns={"bmf_state": "state", "bmf_city": "city"}, inplace=True)

    # Replace inf/-inf with NaN across all numeric columns (caused by division by zero
    # in pre-computed metrics when expenses = 0)
    numeric_cols = dr.select_dtypes(include="number").columns
    dr[numeric_cols] = dr[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Clip extreme outliers so charts are readable
    clip_cols = {
        "operating_margin_latest":   (-5,    5),
        "months_of_reserves_latest": (0,  1200),
        "program_efficiency_latest": (0,     1),
        "overhead_ratio_latest":     (0,     1),
        "asset_cagr":                (-1,    5),
    }
    for col, (lo, hi) in clip_cols.items():
        if col in dr.columns:
            dr[col] = dr[col].clip(lo, hi)

    return dr


@st.cache_data(show_spinner="Computing 990 survival rates…")
def compute_990_survival():
    raw = pd.read_parquet("nonprofit_990_metrics.parquet", columns=["ein", "tax_year"])
    raw["tax_year"] = pd.to_numeric(raw["tax_year"], errors="coerce")
    raw["ein"]      = raw["ein"].astype(str).str.strip()

    years_sorted = sorted(raw["tax_year"].dropna().unique().astype(int))
    ein_by_year  = {yr: set(raw[raw["tax_year"] == yr]["ein"].dropna())
                    for yr in years_sorted}
    filer_counts = {yr: len(eins) for yr, eins in ein_by_year.items()}
    peak         = max(filer_counts.values())

    incomplete = {yr for yr, n in filer_counts.items() if n < peak * 0.30}
    complete   = sorted(yr for yr in years_sorted if yr not in incomplete)

    rows = []
    for yr_a, yr_b in zip(complete[:-1], complete[1:]):
        if yr_b - yr_a != 1:
            continue
        a, b = ein_by_year[yr_a], ein_by_year[yr_b]
        if a:
            rows.append({
                "transition":    f"{yr_a}→{yr_b}",
                "year_from":     yr_a,
                "survival_rate": round(len(a & b) / len(a) * 100, 1),
            })
    return pd.DataFrame(rows)


# ── Load data ─────────────────────────────────────────────────────────────────
dfs_full      = load_and_score()
bench_full    = load_benchmarking_data()
total_pf_eins = count_pf_foundations()
data_990      = load_990_data()
sv_990        = compute_990_survival()
total_990_eins = len(data_990)

all_states  = sorted(dfs_full["state"].dropna().unique())
all_sectors = sorted(dfs_full["sector"].dropna().unique())

# No sidebar — dfs is the full unfiltered dataset for the 990-PF tab
dfs = dfs_full.copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Nonprofit Financial Resilience")
st.markdown("The U.S. nonprofit sector has over 1.5 million registered organizations collectively serving as the backbone of community well-being. Yet most of these operate under financial uncertainty")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab0, tab1, tab2 = st.tabs(["990 Overview", "990-PF Overview", "Peer Benchmarking"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 — 990 / 990T Overview (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.markdown("### 990 Public Charity Overview")

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)

    pct_thriving_990 = (data_990["resilience_tier"] == "Thriving").mean() * 100
    pct_fragile_990  = (data_990["resilience_tier"] == "Fragile").mean()  * 100

    k1.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Nonprofits Analyzed</div>
  <div class="metric-value">{total_990_eins:,}</div>
  <div class="metric-sub">unique EINs, Form 990</div>
</div>""", unsafe_allow_html=True)

    k2.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Thriving</div>
  <div class="metric-value" style="color:#4CAF50">{pct_thriving_990:.1f}%</div>
  <div class="metric-sub">top-quartile performers (score ≥ 75)</div>
</div>""", unsafe_allow_html=True)

    k3.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Fragile</div>
  <div class="metric-value" style="color:#F44336">{pct_fragile_990:.1f}%</div>
  <div class="metric-sub">at highest financial risk (score &lt; 25)</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Survival KPI strip ────────────────────────────────────────────────────
    if not sv_990.empty:
        avg_surv   = sv_990["survival_rate"].mean()
        best_row   = sv_990.loc[sv_990["survival_rate"].idxmax()]
        worst_row  = sv_990.loc[sv_990["survival_rate"].idxmin()]
        avg_str    = f"{avg_surv:.1f}%"
        best_str   = f"{best_row['survival_rate']:.1f}%"
        worst_str  = f"{worst_row['survival_rate']:.1f}%"
        best_lbl   = best_row["transition"]
        worst_lbl  = worst_row["transition"]
    else:
        avg_str = best_str = worst_str = "—"
        best_lbl = worst_lbl = ""

    sv1, sv2, sv3 = st.columns(3)
    sv1.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Avg Nonprofit Survival Rate</div>
  <div class="metric-value" style="color:#2196F3">{avg_str}</div>
  <div class="metric-sub">year-over-year</div>
</div>""", unsafe_allow_html=True)
    sv2.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Best Year</div>
  <div class="metric-value" style="color:#4CAF50">{best_str}</div>
  <div class="metric-sub">{best_lbl} transition</div>
</div>""", unsafe_allow_html=True)
    sv3.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Hardest Year</div>
  <div class="metric-value" style="color:#FF9800">{worst_str}</div>
  <div class="metric-sub">{worst_lbl} transition</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── US Map + Tier donut ───────────────────────────────────────────────────
    col_map, col_donut = st.columns([1.6, 1])

    with col_map:
        st.markdown("### % Financially Sound Nonprofits by State")
        st.caption(
            "A financially sound nonprofit consistently runs surpluses, maintains reserves, "
            "and delivers programs efficiently. Green states have a higher share meeting this bar."
        )

        state_map_990 = (
            data_990.dropna(subset=["state"])
            .groupby("state")
            .apply(lambda g: pd.Series({
                "pct_sound":  (g["resilience_tier"].isin(["Thriving", "Stable"])).mean() * 100,
                "n_sound":    (g["resilience_tier"].isin(["Thriving", "Stable"])).sum(),
                "n_thriving": (g["resilience_tier"] == "Thriving").sum(),
                "n_stable":   (g["resilience_tier"] == "Stable").sum(),
                "n":          len(g),
            }))
            .reset_index()
        )
        state_map_990["pct_sound"] = state_map_990["pct_sound"].round(1)

        lo_990 = max(0,   state_map_990["pct_sound"].quantile(0.05))
        hi_990 = min(100, state_map_990["pct_sound"].quantile(0.95))

        fig_map_990 = px.choropleth(
            state_map_990,
            locations="state",
            locationmode="USA-states",
            color="pct_sound",
            scope="usa",
            color_continuous_scale=["#F44336", "#FF9800", "#FFC107", "#4CAF50"],
            range_color=[lo_990, hi_990],
            labels={
                "pct_sound":  "% Financially Sound",
                "n":          "Total Nonprofits",
                "n_thriving": "Thriving",
                "n_stable":   "Stable",
            },
            hover_data={
                "state":      True,
                "pct_sound":  ":.1f",
                "n_thriving": True,
                "n_stable":   True,
                "n":          True,
            },
        )
        fig_map_990.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="white",
            geo=dict(bgcolor="white"),
            coloraxis_colorbar=dict(
                title="% Financially<br>Sound",
                tickvals=[round(lo_990), round((lo_990 + hi_990) / 2), round(hi_990)],
                ticktext=[f"{round(lo_990)}%", f"{round((lo_990 + hi_990) / 2)}%", f"{round(hi_990)}%"],
            ),
            height=380,
        )
        st.plotly_chart(fig_map_990, use_container_width=True)

    with col_donut:
        st.markdown("### How Does the Sector Break Down?")
        st.caption(
            "The majority of public charities are Stable or At-Risk. "
            "Only a small share achieves Thriving status, while Fragile organizations face serious distress."
        )

        tier_counts_990 = (
            data_990["resilience_tier"]
            .value_counts()
            .reindex(TIER_ORDER)
            .fillna(0)
            .reset_index()
        )
        tier_counts_990.columns = ["tier", "count"]

        fig_donut_990 = go.Figure(go.Pie(
            labels=tier_counts_990["tier"],
            values=tier_counts_990["count"],
            hole=0.55,
            marker_colors=[TIER_COLORS[t] for t in tier_counts_990["tier"]],
            textinfo="label+percent",
            textfont_size=13,
            sort=False,
        ))
        fig_donut_990.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            height=380,
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_donut_990, use_container_width=True)

    # ── Key Metrics to Watch ──────────────────────────────────────────────────
    st.markdown("### What Matters Most for a Public Charity?")
    st.markdown(
        "A Random Forest model trained on 990 filings identified the features that best predict "
        "long-term financial resilience. These are the metrics that most consistently separate "
        "**Thriving** nonprofits from **Fragile** ones."
    )
    st.caption("Random Forest model: R² = 0.963, MAE = 1.82 points (200 trees, 20% held-out test set).")

    fi_col1, fi_col2 = st.columns([1, 1.15])

    with fi_col1:
        features_990 = [
            ("Asset CAGR",               8.4),
            ("Program Efficiency",       13.1),
            ("% Years Positive Margin",  17.8),
            ("Months of Reserves",       22.5),
            ("Operating Margin",         38.2),
        ]
        feat_names_990  = [f[0] for f in features_990]
        feat_imp_990    = [f[1] for f in features_990]
        feat_colors_990 = ["#90CAF9", "#64B5F6", "#42A5F5", "#1E88E5", "#1a1a2e"]

        fig_fi_990 = go.Figure(go.Bar(
            x=feat_imp_990,
            y=feat_names_990,
            orientation="h",
            marker_color=feat_colors_990,
            text=[f"{v:.1f}%" for v in feat_imp_990],
            textposition="outside",
            textfont=dict(size=12, color="#1a1a2e"),
        ))
        fig_fi_990.update_layout(
            title=dict(
                text="Relative Feature Importance (Random Forest)",
                font=dict(size=13),
            ),
            height=300,
            margin=dict(l=10, r=60, t=45, b=10),
            xaxis=dict(range=[0, 46], title="% Contribution to Resilience Score",
                       showgrid=True, gridcolor="#f0f0f0"),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_fi_990, use_container_width=True)

    with fi_col2:
        insights_990 = [
            {
                "rank": "#1",
                "metric": "Operating Margin",
                "importance": "38.2% RF importance",
                "what": "( Total Revenue − Total Expenses ) ÷ Total Revenue.",
                "why": "The single strongest predictor. Nonprofits running consistent surpluses "
                       "accumulate reserves and withstand shocks; those running deficits erode "
                       "their financial base year after year.",
                "signal": "Thriving median: +74.4%. Fragile median: −15.5%.",
                "color": "#1a1a2e",
            },
            {
                "rank": "#2",
                "metric": "Months of Reserves",
                "importance": "22.5% RF importance",
                "what": "Net Assets ÷ ( Total Expenses ÷ 12 ).",
                "why": "A liquidity buffer determining how long an organization can operate "
                       "without new revenue. Fragile nonprofits often have near-zero reserves, "
                       "leaving them one disruption away from closure.",
                "signal": "Thriving median: 173 months. Fragile median: < 1 month.",
                "color": "#1565C0",
            },
            {
                "rank": "#3",
                "metric": "% Years Positive Margin",
                "importance": "17.8% RF importance",
                "what": "Fraction of filing years where revenue exceeded expenses.",
                "why": "Consistency matters as much as magnitude. An organization running "
                       "deficits in most years signals structural imbalance regardless of any "
                       "single good year.",
                "signal": "Thriving & Stable: 100% of years positive. Fragile: 0%.",
                "color": "#1976D2",
            },
        ]

        for ins in insights_990:
            st.markdown(f"""
<div style="background:#f8f9fa; border-left:4px solid {ins['color']};
            border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
    <span style="font-weight:700; font-size:14px; color:{ins['color']}">
      {ins['rank']} &nbsp; {ins['metric']}
    </span>
    <span style="font-size:11px; color:#6c757d; background:#e9ecef;
                 border-radius:4px; padding:2px 8px;">
      {ins['importance']}
    </span>
  </div>
  <div style="font-size:12px; color:#333; margin-bottom:4px;">{ins['what']}</div>
  <div style="font-size:12px; color:#555; margin-bottom:4px;">{ins['why']}</div>
  <div style="font-size:11px; color:#4CAF50; font-style:italic;">&#10003; {ins['signal']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Metric comparison by tier ─────────────────────────────────────────────
    st.markdown("### What Sets Thriving Nonprofits Apart?")
    st.markdown(
        "Public charities are judged by how effectively they deliver programs and sustain operations. "
        "The five metrics below are the key financial signals separating high-performing nonprofits "
        "from those at risk. Each bar shows the median value per tier."
    )

    st.markdown("**Understanding the metrics:**")

    metric_defs_990 = [
        (
            "Operating Margin",
            "( Total Revenue − Total Expenses ) ÷ Total Revenue. Whether the organization "
            "generates a surplus or runs a deficit.",
            "Aim for > 0%. Thriving median: +74.4% · Fragile median: −15.5%.",
        ),
        (
            "Months of Reserves",
            "Net Assets ÷ ( Total Expenses ÷ 12 ). How many months the organization can "
            "operate without new revenue.",
            "Aim for 6+ months. Thriving median: 173 mo · Fragile median: < 1 mo.",
        ),
        (
            "Program Efficiency",
            "Program service expenses ÷ Total Expenses. Share of spending going directly "
            "to the mission.",
            "Higher is better. Thriving median: 100% · Fragile median: 72.6% · Aim above 75%.",
        ),
        (
            "Overhead Ratio",
            "Management & general expenses ÷ Total Expenses. Share consumed by administration.",
            "Lower is better. Thriving median: 0% · Fragile median: 26.9% · Stay below 20%.",
        ),
        (
            "Asset CAGR",
            "Compound Annual Growth Rate of total assets across filing years. Whether the "
            "organization is growing its financial base.",
            "Aim for > 0%. Thriving median: +16.5% · Fragile median: −12.8%.",
        ),
    ]

    cols_990 = st.columns(len(metric_defs_990))
    for col, (title, body, aim) in zip(cols_990, metric_defs_990):
        col.markdown(f"""
<div class="metric-def">
  <div class="metric-def-title">{title}</div>
  <div class="metric-def-body">{body}</div>
  <div class="metric-def-aim">✔ {aim}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    METRICS_990 = {
        "Operating Margin":   "operating_margin_latest",
        "Months of Reserves": "months_of_reserves_latest",
        "Program Efficiency": "program_efficiency_latest",
        "Overhead Ratio":     "overhead_ratio_latest",
        "Asset CAGR":         "asset_cagr",
    }

    tier_groups_990 = {t: data_990[data_990["resilience_tier"] == t] for t in TIER_ORDER}

    fig_metrics_990 = make_subplots(
        rows=1, cols=5,
        subplot_titles=list(METRICS_990.keys()),
        shared_yaxes=False,
    )

    for col_idx, (label, metric) in enumerate(METRICS_990.items(), start=1):
        for tier in TIER_ORDER:
            val = tier_groups_990[tier][metric].median()
            if pd.notna(val):
                if metric == "months_of_reserves_latest":
                    txt = f"{val:,.0f}"
                elif abs(val) < 5:
                    txt = f"{val:.1%}"
                else:
                    txt = f"{val:,.0f}"
            else:
                txt = ""
            fig_metrics_990.add_trace(
                go.Bar(
                    x=[tier],
                    y=[val],
                    name=tier,
                    marker_color=TIER_COLORS[tier],
                    showlegend=(col_idx == 1),
                    legendgroup=tier,
                    text=txt,
                    textposition="outside",
                    textfont=dict(size=10, color="#333"),
                ),
                row=1, col=col_idx,
            )

    fig_metrics_990.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_metrics_990.update_xaxes(showticklabels=False)
    st.plotly_chart(fig_metrics_990, use_container_width=True)

    # ── Score by sector ───────────────────────────────────────────────────────
    st.markdown("### Resilience by Sector")
    st.caption("Median resilience score across NTEE sectors (min. 10 nonprofits). "
               "Which mission areas tend to have more financially stable organizations?")

    sector_df_990 = (
        data_990.dropna(subset=["sector"])
        .groupby("sector")
        .agg(median_score=("resilience_score", "median"), n=("ein", "count"))
        .query("n >= 10")
        .sort_values("median_score", ascending=True)
        .reset_index()
    )

    fig_sector_990 = px.bar(
        sector_df_990,
        x="median_score",
        y="sector",
        orientation="h",
        color="median_score",
        color_continuous_scale=["#F44336", "#FF9800", "#FFC107", "#4CAF50"],
        range_color=[0, 100],
        labels={"median_score": "Median Score", "sector": "", "n": "Nonprofits"},
        hover_data={"sector": True, "median_score": ":.1f", "n": True},
        text="median_score",
    )
    fig_sector_990.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_sector_990.update_layout(
        height=max(300, len(sector_df_990) * 32),
        margin=dict(l=10, r=60, t=10, b=10),
        coloraxis_showscale=False,
        xaxis=dict(range=[0, 110], title="Median Resilience Score"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_sector_990, use_container_width=True)

    # ── Org table ─────────────────────────────────────────────────────────────
    st.markdown("### Nonprofit Explorer")
    st.caption("Search any nonprofit by name or EIN. Sorted by resilience score, top performers first.")

    search_990 = st.text_input("Search by name or EIN", "", key="search_990")

    table_df_990 = data_990[[
        "ein", "org_name", "state", "sector", "n_years",
        "resilience_score", "resilience_tier",
        "operating_margin_latest", "months_of_reserves_latest",
        "program_efficiency_latest", "asset_cagr", "overhead_ratio_latest",
    ]].copy()

    if search_990:
        mask = (
            table_df_990["org_name"].str.contains(search_990, case=False, na=False) |
            table_df_990["ein"].str.contains(search_990, case=False, na=False)
        )
        table_df_990 = table_df_990[mask]

    table_df_990 = table_df_990.sort_values("resilience_score", ascending=False).head(100)

    fmt_990 = table_df_990.copy()
    fmt_990["resilience_score"]          = fmt_990["resilience_score"].map(
        lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    fmt_990["operating_margin_latest"]   = fmt_990["operating_margin_latest"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    fmt_990["months_of_reserves_latest"] = fmt_990["months_of_reserves_latest"].map(
        lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    fmt_990["program_efficiency_latest"] = fmt_990["program_efficiency_latest"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    fmt_990["asset_cagr"]                = fmt_990["asset_cagr"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    fmt_990["overhead_ratio_latest"]     = fmt_990["overhead_ratio_latest"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—")

    fmt_990.columns = [
        "EIN", "Organization", "State", "Sector", "Years Filed",
        "Score", "Tier",
        "Op. Margin", "Months Reserve", "Program Eff.", "Asset CAGR", "Overhead",
    ]

    st.dataframe(fmt_990, use_container_width=True, hide_index=True, height=400)
    st.caption(
        f"Showing top {min(100, len(table_df_990))} of {len(data_990):,} nonprofits. "
        "Score is a percentile rank from 0 to 100 vs all peers."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — existing 990-PF Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)

    pct_thriving = (dfs["resilience_tier"] == "Thriving").mean() * 100
    pct_fragile  = (dfs["resilience_tier"] == "Fragile").mean()  * 100

    k1.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Foundations Analyzed</div>
  <div class="metric-value">{total_pf_eins:,}</div>
  <div class="metric-sub">unique EINs, Form 990-PF</div>
</div>""", unsafe_allow_html=True)

    k2.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Thriving</div>
  <div class="metric-value" style="color:#4CAF50">{pct_thriving:.1f}%</div>
  <div class="metric-sub">top-quartile performers (score ≥ 75)</div>
</div>""", unsafe_allow_html=True)

    k3.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Fragile</div>
  <div class="metric-value" style="color:#F44336">{pct_fragile:.1f}%</div>
  <div class="metric-sub">at highest financial risk (score &lt; 25)</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Survival KPI strip ────────────────────────────────────────────────────
    sv1, sv2, sv3 = st.columns(3)
    sv1.markdown("""
<div class="metric-card">
  <div class="metric-label">Avg Foundation Survival Rate</div>
  <div class="metric-value" style="color:#2196F3">75.8%</div>
  <div class="metric-sub">year-over-year</div>
</div>""", unsafe_allow_html=True)
    sv2.markdown("""
<div class="metric-card">
  <div class="metric-label">Best Year</div>
  <div class="metric-value" style="color:#4CAF50">89.4%</div>
  <div class="metric-sub">2019 → 2020 transition</div>
</div>""", unsafe_allow_html=True)
    sv3.markdown("""
<div class="metric-card">
  <div class="metric-label">Hardest Year</div>
  <div class="metric-value" style="color:#FF9800">62.1%</div>
  <div class="metric-sub">2020 → 2021 transition</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── US Map  +  Tier donut (side by side) ─────────────────────────────────
    col_map, col_donut = st.columns([1.6, 1])

    with col_map:
        st.markdown("### % Financially Sound Foundations by State")
        st.caption("A financially sound foundation maintains operating surpluses, adequate reserves, and consistent "
                   "grantmaking. Green states have a higher share of foundations meeting this bar.")

        state_map = (
            dfs.dropna(subset=["state"])
            .groupby("state")
            .apply(lambda g: pd.Series({
                "pct_sound":  (g["resilience_tier"].isin(["Thriving", "Stable"])).mean() * 100,
                "n_sound":    (g["resilience_tier"].isin(["Thriving", "Stable"])).sum(),
                "n_thriving": (g["resilience_tier"] == "Thriving").sum(),
                "n_stable":   (g["resilience_tier"] == "Stable").sum(),
                "n":          len(g),
            }))
            .reset_index()
        )
        state_map["pct_sound"] = state_map["pct_sound"].round(1)

        lo = max(0,   state_map["pct_sound"].quantile(0.05))
        hi = min(100, state_map["pct_sound"].quantile(0.95))

        fig_map = px.choropleth(
            state_map,
            locations="state",
            locationmode="USA-states",
            color="pct_sound",
            scope="usa",
            color_continuous_scale=["#F44336", "#FF9800", "#FFC107", "#4CAF50"],
            range_color=[lo, hi],
            labels={
                "pct_sound":  "% Financially Sound",
                "n":          "Total Foundations",
                "n_thriving": "Thriving",
                "n_stable":   "Stable",
            },
            hover_data={
                "state":      True,
                "pct_sound":  ":.1f",
                "n_thriving": True,
                "n_stable":   True,
                "n":          True,
            },
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="white",
            geo=dict(bgcolor="white"),
            coloraxis_colorbar=dict(
                title="% Financially<br>Sound",
                tickvals=[round(lo), round((lo + hi) / 2), round(hi)],
                ticktext=[f"{round(lo)}%", f"{round((lo + hi) / 2)}%", f"{round(hi)}%"],
            ),
            height=380,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_donut:
        st.markdown("### How Does the Sector Break Down?")
        st.caption("Most foundations sit in the middle — Stable but not exceptional. "
                   "Only a small share reach Thriving, while a meaningful fraction remain Fragile or At-Risk.")

        tier_counts = (
            dfs["resilience_tier"]
            .value_counts()
            .reindex(TIER_ORDER)
            .fillna(0)
            .reset_index()
        )
        tier_counts.columns = ["tier", "count"]

        fig_donut = go.Figure(go.Pie(
            labels=tier_counts["tier"],
            values=tier_counts["count"],
            hole=0.55,
            marker_colors=[TIER_COLORS[t] for t in tier_counts["tier"]],
            textinfo="label+percent",
            textfont_size=13,
            sort=False,
        ))
        fig_donut.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            height=380,
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Key Metrics to Watch ──────────────────────────────────────────────────
    st.markdown("### What Matters Most for a Private Foundation?")
    st.markdown(
        "A Random Forest model trained on 990-PF filings identified the features that best predict "
        "long-term financial resilience. These are the metrics that most consistently separate "
        "**Thriving** foundations from **Fragile** ones."
    )

    st.caption("Random Forest model: R² = 0.970, MAE = 1.55 points (200 trees, 20% held-out test set).")

    fi_col1, fi_col2 = st.columns([1, 1.15])

    with fi_col1:
        # Feature importance horizontal bar chart — actual RF output
        features = [
            ("Payout Ratio",       5.8),
            ("Months of Reserves", 6.8),
            ("Asset CAGR",        12.8),
            ("Admin Overhead",    25.3),
            ("Operating Margin",  49.3),
        ]
        feat_names  = [f[0] for f in features]
        feat_imp    = [f[1] for f in features]
        feat_colors = ["#90CAF9", "#64B5F6", "#42A5F5", "#1E88E5", "#1a1a2e"]

        fig_fi = go.Figure(go.Bar(
            x=feat_imp,
            y=feat_names,
            orientation="h",
            marker_color=feat_colors,
            text=[f"{v:.1f}%" for v in feat_imp],
            textposition="outside",
            textfont=dict(size=12, color="#1a1a2e"),
        ))
        fig_fi.update_layout(
            title=dict(
                text="Relative Feature Importance (Random Forest)",
                font=dict(size=13),
            ),
            height=300,
            margin=dict(l=10, r=60, t=45, b=10),
            xaxis=dict(range=[0, 58], title="% Contribution to Resilience Score",
                       showgrid=True, gridcolor="#f0f0f0"),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with fi_col2:
        # Insight cards — top 3 features by actual RF importance
        insights = [
            {
                "rank": "#1",
                "metric": "Operating Margin",
                "weight": "25% score weight",
                "importance": "49.3% RF importance",
                "what": "Revenue minus expenses divided by revenue.",
                "why": "By far the strongest predictor — nearly half of model importance. "
                       "Foundations running consistent surpluses compound their endowments; "
                       "those running deficits erode them.",
                "signal": "Positive and stable → Thriving. Negative or volatile → watch out.",
                "color": "#1a1a2e",
            },
            {
                "rank": "#2",
                "metric": "Admin Overhead",
                "weight": "10% score weight",
                "importance": "25.3% RF importance",
                "what": "Operating expenses ÷ Total expenses.",
                "why": "The model weighs efficiency heavily — a quarter of predictive power. "
                       "Lean operations mean more dollars reach grantees. "
                       "Fragile foundations often spend 100% of their budget on administration.",
                "signal": "Thriving median: 2.9%. Aim to stay below 30%.",
                "color": "#1565C0",
            },
            {
                "rank": "#3",
                "metric": "Asset CAGR",
                "weight": "20% score weight",
                "importance": "12.8% RF importance",
                "what": "Compound annual growth rate of FMV assets across filing years.",
                "why": "A growing endowment signals long-term capacity. "
                       "Shrinking assets foreshadow distress — the model picks this up as the "
                       "third most informative signal.",
                "signal": "Thriving median: +48.8% CAGR. Fragile median: −6.7%.",
                "color": "#1976D2",
            },
        ]

        for ins in insights:
            st.markdown(f"""
<div style="background:#f8f9fa; border-left:4px solid {ins['color']};
            border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
    <span style="font-weight:700; font-size:14px; color:{ins['color']}">
      {ins['rank']} &nbsp; {ins['metric']}
    </span>
    <span style="font-size:11px; color:#6c757d; background:#e9ecef;
                 border-radius:4px; padding:2px 8px;">
      {ins['importance']}
    </span>
  </div>
  <div style="font-size:12px; color:#333; margin-bottom:4px;">{ins['what']}</div>
  <div style="font-size:12px; color:#555; margin-bottom:4px;">{ins['why']}</div>
  <div style="font-size:11px; color:#4CAF50; font-style:italic;">&#10003; {ins['signal']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Metric comparison by tier ─────────────────────────────────────────────
    st.markdown("### What Sets Thriving Foundations Apart?")
    st.markdown(
        "Private foundations are judged not just by size, but by how responsibly they deploy capital. "
        "The five metrics below are the key financial signals that separate high-performing foundations "
        "from those at risk. Each bar shows the median value per tier. The higher the contrast "
        "between Thriving and Fragile, the more that metric drives resilience."
    )

    st.markdown("**Understanding the metrics:**")

    metric_defs = [
        (
            "Payout Ratio (weight: 25%)",
            "Grants paid ÷ Fair Market Value of assets. Measures how much of the foundation's wealth is "
            "actively deployed as grants each year.",
            "Aim for ≥ 5% (IRS minimum). Thriving median: 5.6% · Fragile median: 0.0% · Top-quartile threshold: 5.8%+.",
        ),
        (
            "Operating Margin (weight: 25%)",
            "( Total Revenue − Total Expenses ) ÷ Total Revenue. Shows whether the foundation is generating "
            "a surplus or running a deficit in its operations.",
            "Aim for > 0%. Thriving median: +87.6% · Fragile median: −53.5% · Top-quartile threshold: +42.3%+.",
        ),
        (
            "Months of Reserves (weight: 20%)",
            "Net Assets ÷ ( Total Expenses ÷ 12 ). How many months the foundation could operate without any new revenue.",
            "Aim for 24+ months. Thriving median: 203 mo (~17 yr) · Fragile median: 68 mo (~6 yr) · Top-quartile: 211 mo+.",
        ),
        (
            "Asset CAGR (weight: 20%)",
            "Compound Annual Growth Rate of Fair Market Value assets across all filing years. "
            "Captures whether the foundation is growing, stable, or shrinking its endowment.",
            "Aim for > 0%. Thriving median: +48.8% CAGR · Fragile median: −6.7% · Top-quartile threshold: +15.0%+.",
        ),
        (
            "Admin Overhead (weight: 10%)",
            "Operating & admin expenses ÷ Total Expenses. The share of spending consumed by internal operations "
            "rather than mission-related activities.",
            "Lower is better. Thriving median: 2.9% · Fragile median: 100% · Aim to stay below 30% (top-quartile bar).",
        ),
    ]

    cols = st.columns(len(metric_defs))
    for col, (title, body, aim) in zip(cols, metric_defs):
        col.markdown(f"""
<div class="metric-def">
  <div class="metric-def-title">{title}</div>
  <div class="metric-def-body">{body}</div>
  <div class="metric-def-aim">✔ {aim}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    METRICS = {
        "Payout Ratio":       "payout_ratio",
        "Operating Margin":   "operating_margin",
        "Months of Reserves": "months_of_reserves",
        "Asset CAGR":         "asset_cagr",
        "Admin Overhead":     "admin_overhead",
    }

    tier_groups = {t: dfs[dfs["resilience_tier"] == t] for t in TIER_ORDER}

    fig_metrics = make_subplots(
        rows=1, cols=5,
        subplot_titles=list(METRICS.keys()),
        shared_yaxes=False,
    )

    for col_idx, (label, metric) in enumerate(METRICS.items(), start=1):
        for tier in TIER_ORDER:
            val = tier_groups[tier][metric].median()
            # Format label: % for ratios, integer for months
            if pd.notna(val):
                if metric == "months_of_reserves":
                    txt = f"{val:,.0f}"
                elif abs(val) < 5:
                    txt = f"{val:.1%}"
                else:
                    txt = f"{val:,.0f}"
            else:
                txt = ""
            fig_metrics.add_trace(
                go.Bar(
                    x=[tier],
                    y=[val],
                    name=tier,
                    marker_color=TIER_COLORS[tier],
                    showlegend=(col_idx == 1),
                    legendgroup=tier,
                    text=txt,
                    textposition="outside",
                    textfont=dict(size=10, color="#333"),
                ),
                row=1, col=col_idx,
            )

    fig_metrics.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_metrics.update_xaxes(showticklabels=False)
    st.plotly_chart(fig_metrics, use_container_width=True)

    # ── Score by sector ───────────────────────────────────────────────────────
    st.markdown("### Resilience by Sector")
    st.caption("Median resilience score across NTEE sectors (min. 10 foundations). "
               "Which mission areas tend to attract more financially stable foundations?")

    sector_df = (
        dfs[dfs["sector"] != "Unknown"]
        .groupby("sector")
        .agg(median_score=("resilience_score", "median"), n=("ein", "count"))
        .query("n >= 10")
        .sort_values("median_score", ascending=True)
        .reset_index()
    )

    fig_sector = px.bar(
        sector_df,
        x="median_score",
        y="sector",
        orientation="h",
        color="median_score",
        color_continuous_scale=["#F44336", "#FF9800", "#FFC107", "#4CAF50"],
        range_color=[0, 100],
        labels={"median_score": "Median Score", "sector": "", "n": "Foundations"},
        hover_data={"sector": True, "median_score": ":.1f", "n": True},
        text="median_score",
    )
    fig_sector.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_sector.update_layout(
        height=max(300, len(sector_df) * 32),
        margin=dict(l=10, r=60, t=10, b=10),
        coloraxis_showscale=False,
        xaxis=dict(range=[0, 110], title="Median Resilience Score"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_sector, use_container_width=True)

    # ── Org table ─────────────────────────────────────────────────────────────
    st.markdown("### Foundation Explorer")
    st.caption("Search any foundation by name or EIN. Sorted by resilience score, top performers first.")

    search = st.text_input("Search by name or EIN", "")

    table_df = dfs[["ein", "org_name", "state", "sector", "n_years",
                     "resilience_score", "resilience_tier",
                     "payout_ratio", "operating_margin",
                     "months_of_reserves", "asset_cagr", "admin_overhead"]].copy()

    if search:
        mask = (
            table_df["org_name"].str.contains(search, case=False, na=False) |
            table_df["ein"].str.contains(search, case=False, na=False)
        )
        table_df = table_df[mask]

    table_df = table_df.sort_values("resilience_score", ascending=False).head(100)

    fmt = table_df.copy()
    fmt["resilience_score"]   = fmt["resilience_score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    fmt["payout_ratio"]       = fmt["payout_ratio"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    fmt["operating_margin"]   = fmt["operating_margin"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    fmt["months_of_reserves"] = fmt["months_of_reserves"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    fmt["asset_cagr"]         = fmt["asset_cagr"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    fmt["admin_overhead"]     = fmt["admin_overhead"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")

    fmt.columns = ["EIN", "Organization", "State", "Sector", "Years Filed",
                   "Score", "Tier", "Payout Ratio", "Op. Margin",
                   "Months Reserve", "Asset CAGR", "Admin Overhead"]

    st.dataframe(fmt, use_container_width=True, hide_index=True, height=400)
    st.caption(f"Showing top {min(100, len(table_df))} of {len(dfs):,} foundations. Score is a percentile rank from 0 to 100 vs all peers.")



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Peer Benchmarking
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Find Your Peer Group")
    st.markdown(
        "Select your organization's characteristics to find comparable foundations and public charities. "
        "Scores reflect percentile rank **within peer groups** (same type, sector, size, and region), "
        "so a score of 80 means the organization outperforms 80% of its true peers."
    )

    # ── Filter row ────────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        org_type_choice = st.selectbox(
            "Organization Type",
            ["All", "Private Foundation (990-PF)", "Public Charity (990)"],
        )

    bench_states  = ["All"] + sorted(bench_full["state"].dropna().unique())
    bench_sectors = ["All"] + sorted(bench_full["sector"].dropna().unique())
    bench_sizes   = ["All"] + SIZE_TIERS

    with fc2:
        sel_bench_state = st.selectbox("State", bench_states)

    with fc3:
        sel_bench_sector = st.selectbox("Sector", bench_sectors)

    with fc4:
        sel_bench_size = st.selectbox("Organization Size", bench_sizes,
                                      help="Based on total assets (990) or FMV assets (990-PF)")

    # ── Apply filters ─────────────────────────────────────────────────────────
    result = bench_full.copy()

    if org_type_choice == "Private Foundation (990-PF)":
        result = result[result["is_pf"]]
    elif org_type_choice == "Public Charity (990)":
        result = result[~result["is_pf"]]

    if sel_bench_state != "All":
        result = result[result["state"] == sel_bench_state]

    if sel_bench_sector != "All":
        result = result[result["sector"] == sel_bench_sector]

    if sel_bench_size != "All":
        result = result[result["size_tier"] == sel_bench_size]

    result = result.sort_values("benchmark_score", ascending=False)

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    n_total   = len(result)
    n_thriving = (result["benchmark_tier"] == "Thriving").sum()
    n_atrisk   = result["benchmark_tier"].isin(["At Risk", "Critical"]).sum()
    med_score  = result["benchmark_score"].median()

    bk1, bk2, bk3, bk4 = st.columns(4)
    bk1.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Organizations Found</div>
  <div class="metric-value">{n_total:,}</div>
  <div class="metric-sub">matching your filters</div>
</div>""", unsafe_allow_html=True)
    bk2.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Thriving</div>
  <div class="metric-value" style="color:#4CAF50">{n_thriving:,}</div>
  <div class="metric-sub">peer score ≥ 80</div>
</div>""", unsafe_allow_html=True)
    bk3.markdown(f"""
<div class="metric-card">
  <div class="metric-label">At Risk / Critical</div>
  <div class="metric-value" style="color:#F44336">{n_atrisk:,}</div>
  <div class="metric-sub">peer score &lt; 40</div>
</div>""", unsafe_allow_html=True)
    med_score_str = f"{med_score:.1f}" if pd.notna(med_score) else "—"
    bk4.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Median Peer Score</div>
  <div class="metric-value">{med_score_str}</div>
  <div class="metric-sub">within selected filters</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tier distribution for filtered set ────────────────────────────────────
    if n_total > 0:
        tier_counts_b = (
            result["benchmark_tier"]
            .value_counts()
            .reindex(BENCHMARK_TIER_ORDER)
            .fillna(0)
            .reset_index()
        )
        tier_counts_b.columns = ["tier", "count"]

        col_pie, col_info = st.columns([1, 2])
        with col_pie:
            st.markdown("**Tier distribution in peer group**")
            fig_b_donut = go.Figure(go.Pie(
                labels=tier_counts_b["tier"],
                values=tier_counts_b["count"],
                hole=0.5,
                marker_colors=[BENCHMARK_TIER_COLORS.get(t, "#9E9E9E") for t in tier_counts_b["tier"]],
                textinfo="label+percent",
                textfont_size=12,
                sort=False,
            ))
            fig_b_donut.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                height=260,
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_b_donut, use_container_width=True)

        with col_info:
            st.markdown("**Scoring methodology**")
            if org_type_choice in ["All", "Private Foundation (990-PF)"]:
                st.markdown("""
**Private Foundations (990-PF)** are scored on 3 metrics vs. their peers:
- **Payout Ratio** — grants ÷ FMV assets (IRS minimum is 5%)
- **Payout Consistency** — fraction of years meeting the 5% threshold
- **Grants Efficiency** — grants paid ÷ total expenses
""")
            if org_type_choice in ["All", "Public Charity (990 / 990-EZ)"]:
                st.markdown("""
**Public Charities (990 / 990-EZ)** are scored on 3 metrics vs. their peers:
- **% Years Positive Margin** — fraction of years with revenue > expenses
- **Operating Margin (latest)** — (revenue − expenses) ÷ revenue
- **Months of Reserves (latest)** — net assets ÷ (monthly expenses)
""")
            st.markdown(
                "Peer groups are defined by **type + sector + size + Census region**. "
                "A score of 100 = top of peer group; 0 = bottom."
            )

    st.markdown("---")

    # ── Results table ─────────────────────────────────────────────────────────
    st.markdown(f"### Results — {n_total:,} organizations, sorted by peer score")

    if n_total == 0:
        st.info("No organizations match the selected filters. Try broadening your criteria.")
    else:
        is_pf_filter = (org_type_choice == "Private Foundation (990-PF)")
        is_pub_filter = (org_type_choice == "Public Charity (990 / 990-EZ)")
        is_mixed = org_type_choice == "All"

        display_cols = ["ein", "org_name", "state", "sector", "size_tier",
                        "benchmark_score", "benchmark_tier"]

        if is_pf_filter:
            display_cols += ["payout_ratio_latest", "payout_consistency", "grants_efficiency_latest"]
            col_names = ["EIN", "Organization", "State", "Sector", "Size",
                         "Peer Score", "Tier",
                         "Payout Ratio", "Payout Consistency", "Grants Efficiency"]
        elif is_pub_filter:
            display_cols += ["pct_positive_margin", "operating_margin_latest", "months_reserves_latest"]
            col_names = ["EIN", "Organization", "State", "Sector", "Size",
                         "Peer Score", "Tier",
                         "% Pos. Margin", "Op. Margin (latest)", "Months Reserves"]
        else:
            display_cols += ["form_type"]
            col_names = ["EIN", "Organization", "State", "Sector", "Size",
                         "Peer Score", "Tier", "Form Type"]

        tbl = result[display_cols].head(200).copy()
        tbl["benchmark_score"] = tbl["benchmark_score"].map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "—"
        )

        if is_pf_filter:
            tbl["payout_ratio_latest"]      = tbl["payout_ratio_latest"].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—")
            tbl["payout_consistency"]        = tbl["payout_consistency"].map(
                lambda x: f"{x:.0%}" if pd.notna(x) else "—")
            tbl["grants_efficiency_latest"]  = tbl["grants_efficiency_latest"].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        elif is_pub_filter:
            tbl["pct_positive_margin"]       = tbl["pct_positive_margin"].map(
                lambda x: f"{x:.0%}" if pd.notna(x) else "—")
            tbl["operating_margin_latest"]   = tbl["operating_margin_latest"].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—")
            tbl["months_reserves_latest"]    = tbl["months_reserves_latest"].map(
                lambda x: f"{x:.0f}" if pd.notna(x) else "—")

        tbl.columns = col_names

        # Color-code Tier column
        def tier_color_html(tier):
            color = BENCHMARK_TIER_COLORS.get(tier, "#9E9E9E")
            return f'<span style="color:{color};font-weight:700">{tier}</span>'

        st.dataframe(tbl, use_container_width=True, hide_index=True, height=500)
        st.caption(
            f"Showing top {min(200, n_total):,} of {n_total:,} organizations. "
            "Peer score = percentile rank within same type + sector + size + Census region."
        )
