"""
Beyond the Red Flags — NSE Financial Distress Dashboard
========================================================
Group 12 | NMIMS M.Sc. Statistics & Data Science
Supervisor: Mr. Pratik Desai

Run:  streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Financial Distress | Group 12",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
# Deep navy + coral red + teal + warm gold + slate
C = {
    "blue": "#2E6BE6",
    "red": "#E05C3A",
    "navy":   "#1B2A4A",
    "navy2":  "#243554",
    "blue":   "#2E6BE6",
    "blue_l": "#5B8FEF",
    "coral":  "#E05C3A",
    "coral_l":"#F08C75",
    "teal":   "#1DA882",
    "teal_l": "#5DC7A6",
    "gold":   "#F0A500",
    "gold_l": "#F7C84A",
    "slate":  "#455A7A",
    "slate_l":"#7A93B5",
    "bg":     "#0E1621",
    "card":   "#182336",
    "card2":  "#1E2E45",
    "text":   "#E8EDF5",
    "muted":  "#8899B5",
    "border": "#2A3F5F",
}

HEALTHY_COL  = C["teal"]
DISTRESS_COL = C["coral"]
GREY_COL     = C["gold"]

def hex_to_rgba(hex_color, alpha=0.15):
    """Converts a hex color string to an rgba string for Plotly."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif;
      background-color: {C['bg']};
      color: {C['text']};
  }}
  .stApp {{ background-color: {C['bg']}; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
      background: linear-gradient(180deg, {C['navy']} 0%, {C['navy2']} 100%);
      border-right: 1px solid {C['border']};
  }}
  section[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}

  /* Metric cards */
  [data-testid="stMetric"] {{
      background: {C['card']};
      border: 1px solid {C['border']};
      border-radius: 12px;
      padding: 16px 20px;
  }}
  [data-testid="stMetricValue"] {{ color: {C['blue_l']} !important; font-weight: 700; font-size: 2rem !important; }}
  [data-testid="stMetricLabel"] {{ color: {C['muted']} !important; font-size: 0.8rem !important; }}
  [data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}

  /* Section headers */
  h1 {{ color: {C['text']}; font-weight: 700; letter-spacing: -0.5px; }}
  h2 {{ color: {C['blue_l']}; font-weight: 600; }}
  h3 {{ color: {C['teal_l']}; font-weight: 600; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
      background: {C['card']};
      border-radius: 12px;
      padding: 4px;
      gap: 2px;
      border: 1px solid {C['border']};
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 8px;
      color: {C['muted']};
      font-weight: 500;
      padding: 8px 20px;
  }}
  .stTabs [aria-selected="true"] {{
      background: {C['blue']} !important;
      color: white !important;
  }}

  /* Selectbox / slider */
  .stSelectbox > div, .stMultiSelect > div {{
      background: {C['card2']};
      border: 1px solid {C['border']};
      border-radius: 8px;
  }}

  /* Dataframe */
  .stDataFrame {{ border: 1px solid {C['border']}; border-radius: 10px; }}

  /* Info / warning boxes */
  .info-box {{
      background: {C['card2']};
      border-left: 4px solid {C['blue']};
      border-radius: 0 10px 10px 0;
      padding: 14px 18px;
      margin: 12px 0;
      font-size: 0.9rem;
      color: {C['text']};
  }}
  .warn-box {{
      background: {C['card2']};
      border-left: 4px solid {C['coral']};
      border-radius: 0 10px 10px 0;
      padding: 14px 18px;
      margin: 12px 0;
      font-size: 0.9rem;
  }}
  .success-box {{
      background: {C['card2']};
      border-left: 4px solid {C['teal']};
      border-radius: 0 10px 10px 0;
      padding: 14px 18px;
      margin: 12px 0;
      font-size: 0.9rem;
  }}
  .stat-card {{
      background: {C['card']};
      border: 1px solid {C['border']};
      border-radius: 12px;
      padding: 18px 22px;
      text-align: center;
  }}
  .stat-num {{ font-size: 2rem; font-weight: 700; color: {C['blue_l']}; }}
  .stat-lbl {{ font-size: 0.8rem; color: {C['muted']}; margin-top: 4px; }}
  .divider {{ border-top: 1px solid {C['border']}; margin: 24px 0; }}
  .badge-distressed {{
      background: {C['coral']}22;
      color: {C['coral']};
      border: 1px solid {C['coral']}44;
      border-radius: 20px;
      padding: 2px 10px;
      font-size: 0.75rem;
      font-weight: 600;
  }}
  .badge-healthy {{
      background: {C['teal']}22;
      color: {C['teal']};
      border: 1px solid {C['teal']}44;
      border-radius: 20px;
      padding: 2px 10px;
      font-size: 0.75rem;
      font-weight: 600;
  }}
  .formula-box {{
      background: {C['navy']};
      border: 1px solid {C['blue']}55;
      border-radius: 10px;
      padding: 16px 20px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.95rem;
      color: {C['gold_l']};
      text-align: center;
      margin: 12px 0;
  }}
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    raw   = pd.read_csv("C:/Users/hardi/Downloads/nsestockhistoricalratios.csv",    low_memory=False)
    model = pd.read_csv("C:/Users/hardi/Downloads/Documents/nse_altman_model_ready.csv", low_memory=False)

    for c in raw.columns:
        if c not in ["Year", "Stock"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    for c in model.columns:
        if c not in ["Year", "Stock", "Altman_Zone"]:
            model[c] = pd.to_numeric(model[c], errors="coerce")

    raw["Year"]   = raw["Year"].astype(int)
    model["Year"] = model["Year"].astype(int)

    # Merge distress label into raw so the EDA comparison tab
    # always uses genuine unscaled ratios with correct class labels,
    # regardless of which zone filter the user has selected.
    label_cols = ["Year", "Stock", "Financial_Distress", "Altman_Zone", "Z_Score"]
    raw_labeled = raw.merge(
        model[label_cols], on=["Year", "Stock"], how="inner"
    )
    return raw, model, raw_labeled


@st.cache_data(show_spinner=False)
def build_tte(model):
    """Build time-to-event dataset for survival analysis."""
    records = []
    for stock, grp in model.groupby("Stock"):
        grp = grp.sort_values("Year").reset_index(drop=True)
        if grp["Financial_Distress"].iloc[0] == 1:
            continue
        first_y  = grp["Year"].iloc[0]
        last_y   = grp["Year"].iloc[-1]
        dist_rows = grp[grp["Financial_Distress"] == 1]
        if len(dist_rows) > 0:
            ev_y     = dist_rows["Year"].iloc[0]
            duration = max(ev_y - first_y, 1)
            event    = 1
        else:
            duration = max(last_y - first_y, 1)
            event    = 0
        row = {
            "Stock":    stock,
            "Duration": duration,
            "Event":    event,
            "Z_Score":  grp["Z_Score"].iloc[0],
        }
        for col in ["Current Ratio","Net Profit Margin","Income Quality",
                    "Interest Coverage","Total Debt To Capitalization",
                    "Long Term Debt To Capitalization"]:
            if col in grp.columns:
                row[col] = grp[col].iloc[0]
        records.append(row)
    return pd.DataFrame(records)


# ── Plot theme ─────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter", color=C["text"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"], borderwidth=1),
    xaxis=dict(gridcolor=C["border"], linecolor=C["border"], zerolinecolor=C["border"]),
    yaxis=dict(gridcolor=C["border"], linecolor=C["border"], zerolinecolor=C["border"]),
    margin=dict(l=20, r=20, t=40, b=20),
)

def style(fig, title="", h=400):
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, font=dict(size=14, color=C["text"])), height=h)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 10px 0 20px;'>
      <div style='font-size:2.2rem;'>📉</div>
      <div style='font-weight:700; font-size:1.05rem; color:{C["text"]}; margin-top:6px;'>
        Beyond the Red Flags
      </div>
      <div style='font-size:0.75rem; color:{C["muted"]}; margin-top:4px;'>
        NSE Financial Distress · Group 12
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='color:{C['muted']};font-size:0.75rem;font-weight:600;letter-spacing:1px;'>NAVIGATION</div>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🏠  Overview",
         "🔍  EDA & Filters",
         "📊  Z-Score Analysis",
         "🤖  ML Model Results",
         "⏳  Survival Analysis",
         "🔮  Firm Risk Lookup"],
        label_visibility="collapsed",
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{C['muted']};font-size:0.75rem;font-weight:600;letter-spacing:1px;'>GLOBAL FILTERS</div>", unsafe_allow_html=True)

    with st.spinner("Loading data…"):
        raw, model, raw_labeled = load_data()
        tte                      = build_tte(model)

    year_range = st.slider(
        "Year Range",
        int(model["Year"].min()),
        int(model["Year"].max()),
        (int(model["Year"].min()), int(model["Year"].max())),
    )

    all_stocks = sorted(model["Stock"].unique())
    distressed_stocks = sorted(model[model["Financial_Distress"]==1]["Stock"].unique())
    filter_stocks = st.multiselect(
        "Filter Stocks (leave blank = all)",
        options=all_stocks,
        default=[],
        placeholder="Search stocks…",
    )

    zone_filter = st.multiselect(
        "Altman Zone",
        options=["Healthy","Distressed"],
        default=["Healthy","Distressed"],
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.72rem; color:{C['muted']}; line-height:1.6;'>
      <b style='color:{C['text']}'>Dataset</b><br>
      1,797 NSE Stocks · 2012–2022<br>
      19,767 firm-year observations<br>
      98 financial ratio features<br><br>
      <b style='color:{C['text']}'>Label</b><br>
      Revised Altman Z*-Score (1983)<br>
      Distressed: Z* &lt; 1.10<br>
      Healthy: Z* &gt; 2.60
    </div>
    """, unsafe_allow_html=True)


# ── Apply global filters ───────────────────────────────────────────────────────
mdf = model[
    (model["Year"] >= year_range[0]) &
    (model["Year"] <= year_range[1]) &
    (model["Altman_Zone"].isin(zone_filter))
].copy()
if filter_stocks:
    mdf = mdf[mdf["Stock"].isin(filter_stocks)]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown(f"""
    <div style='padding: 20px 0 10px;'>
      <h1 style='font-size:2rem; margin:0;'>Beyond the Red Flags</h1>
      <div style='color:{C["muted"]}; font-size:1rem; margin-top:6px;'>
        Real-Time Financial Distress Forecasting for NSE-Listed Firms &nbsp;·&nbsp; 2012–2022
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # KPI row
    n_total   = model["Stock"].nunique()
    n_dist    = model[model["Financial_Distress"]==1]["Stock"].nunique()
    n_obs     = len(model)
    dist_rate = model["Financial_Distress"].mean() * 100
    n_events  = int(tte["Event"].sum())
    c_idx     = 0.706

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Unique Firms",     f"{n_total:,}")
    c2.metric("Firm-Year Obs.",   f"{n_obs:,}")
    c3.metric("Distressed Firms", f"{n_dist:,}",  delta=f"{n_dist/n_total*100:.1f}% of all firms")
    c4.metric("Distress Rate",    f"{dist_rate:.1f}%", delta="firm-year level")
    c5.metric("TTE Events",       f"{n_events}",  delta="distress episodes")
    c6.metric("Cox C-Index",      f"{c_idx}",     delta="model discrimination")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3,2])

    with col_l:
        st.subheader("Project Overview")
        st.markdown(f"""
        <div class='info-box'>
        This dashboard accompanies the research treatise <b>"Beyond the Red Flags"</b> submitted by Group 12 to
        SVKM's NMIMS for the M.Sc. Statistics &amp; Data Science programme.
        <br><br>
        The project replicates and extends <b>Lokanan &amp; Ramzan (2024)</b> — an ML-based financial distress
        prediction study for TSX-listed firms — to the Indian NSE context using a richer panel dataset and
        an additional survival analysis dimension.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='formula-box'>
          Z* = 3.25 + 6.56·X₁ + 3.26·X₂ + 6.72·X₃ + 1.05·X₄
          <div style='font-size:0.75rem; color:{C["muted"]}; margin-top:8px; font-family:Inter;'>
            X₁=WC/TA &nbsp;·&nbsp; X₂=ROA &nbsp;·&nbsp; X₃=EBIT/TA &nbsp;·&nbsp; X₄=MktCap/TL
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Zone thresholds
        zc1, zc2, zc3 = st.columns(3)
        with zc1:
            st.markdown(f"""<div class='stat-card'>
              <div class='stat-num' style='color:{DISTRESS_COL};'>Z* &lt; 1.10</div>
              <div class='stat-lbl'>Distressed Zone</div></div>""", unsafe_allow_html=True)
        with zc2:
            st.markdown(f"""<div class='stat-card'>
              <div class='stat-num' style='color:{GREY_COL};'>1.10–2.60</div>
              <div class='stat-lbl'>Grey Zone (excluded)</div></div>""", unsafe_allow_html=True)
        with zc3:
            st.markdown(f"""<div class='stat-card'>
              <div class='stat-num' style='color:{HEALTHY_COL};'>Z* &gt; 2.60</div>
              <div class='stat-lbl'>Healthy Zone</div></div>""", unsafe_allow_html=True)

    with col_r:
        # Zone donut
        zone_counts = model["Altman_Zone"].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=zone_counts.index.tolist(),
            values=zone_counts.values.tolist(),
            hole=0.6,
            marker=dict(colors=[HEALTHY_COL, DISTRESS_COL]),
            textinfo="percent+label",
            textfont=dict(color=C["text"], size=13),
            hovertemplate="%{label}<br>%{value:,} firm-years<br>%{percent}<extra></extra>",
        ))
        fig_donut.update_layout(**PLOT_LAYOUT, height=280,
                                title=dict(text="Firm-Year Distribution", font=dict(size=13, color=C["text"])),
                                annotations=[dict(text="15,513<br>total", x=0.5, y=0.5,
                                                  font=dict(size=14, color=C["text"]), showarrow=False)])
        st.plotly_chart(fig_donut, use_container_width=True)

        # ML results summary table
        st.subheader("ML Results at a Glance")
        ml_df = pd.DataFrame({
            "Model":        ["XGBoost","Decision Tree","Random Forest","ANN","SVM","Logistic Reg."],
            "Test Acc":     ["96.85%","96.0%","90.0%","94.19%","84.0%","81.0%"],
            "F1-Score":     ["0.70","0.66","0.44","0.57","0.32","0.26"],
            "AUC":          ["0.99","0.91","0.97","0.97","0.90","0.88"],
            "Rank":         ["🥇","🥈","3rd","4th","5th","6th"],
        })
        st.dataframe(ml_df.set_index("Rank"), use_container_width=True, height=230)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Timeline: distress events per year
    st.subheader("Distress Events Over Time")
    yr_dist = model.groupby("Year")["Financial_Distress"].agg(["sum","count"]).reset_index()
    yr_dist.columns = ["Year","Distressed","Total"]
    yr_dist["Healthy"] = yr_dist["Total"] - yr_dist["Distressed"]
    yr_dist["Rate"]    = yr_dist["Distressed"] / yr_dist["Total"] * 100

    fig_time = make_subplots(specs=[[{"secondary_y": True}]])
    fig_time.add_trace(go.Bar(x=yr_dist["Year"], y=yr_dist["Healthy"],   name="Healthy",    marker_color=HEALTHY_COL,  opacity=0.85), secondary_y=False)
    fig_time.add_trace(go.Bar(x=yr_dist["Year"], y=yr_dist["Distressed"],name="Distressed", marker_color=DISTRESS_COL, opacity=0.85), secondary_y=False)
    fig_time.add_trace(go.Scatter(x=yr_dist["Year"], y=yr_dist["Rate"], name="Distress %", mode="lines+markers",
                                   line=dict(color=C["gold"], width=2.5, dash="dot"), marker=dict(size=7, color=C["gold"])), secondary_y=True)
    fig_time.update_layout(**PLOT_LAYOUT, barmode="stack", height=350,
                           title=dict(text="Firm-Year Status by Year (stacked) + Distress Rate", font=dict(size=13, color=C["text"])))
    fig_time.update_yaxes(title_text="Firm-Years", secondary_y=False, gridcolor=C["border"])
    fig_time.update_yaxes(title_text="Distress Rate (%)", secondary_y=True, gridcolor=C["border"])
    st.plotly_chart(fig_time, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & FILTERS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  EDA & Filters":
    st.markdown("## Exploratory Data Analysis")
    st.markdown(f"<div style='color:{C['muted']};'>Filtered dataset: **{len(mdf):,}** firm-years · **{mdf['Stock'].nunique():,}** unique stocks</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    NUM_COLS = [c for c in mdf.columns if c not in ["Year","Stock","Altman_Zone","Financial_Distress","Z_Score"]]

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Ratio Distributions", "🔥 Correlation Heatmap", "📉 Ratio Comparison", "📋 Data Explorer"])

    # TAB 1 — Distributions
    with tab1:
        c1, c2 = st.columns([2,1])
        with c1:
            ratio = st.selectbox("Select financial ratio", NUM_COLS,
                                  index=NUM_COLS.index("Net Profit Margin") if "Net Profit Margin" in NUM_COLS else 0)
        with c2:
            n_bins = st.slider("Bins", 20, 100, 40)

        col_data = mdf[["Financial_Distress", ratio]].dropna()
        col_data["Status"] = col_data["Financial_Distress"].map({0:"Healthy",1:"Distressed"})

        p5, p95 = col_data[ratio].quantile(0.05), col_data[ratio].quantile(0.95)
        clipped = col_data[ratio].clip(p5, p95)

        fig_hist = px.histogram(
            col_data.assign(**{ratio: clipped}),
            x=ratio, color="Status", nbins=n_bins,
            color_discrete_map={"Healthy": HEALTHY_COL, "Distressed": DISTRESS_COL},
            opacity=0.8, barmode="overlay",
            template="plotly_dark",
        )
        style(fig_hist, f"Distribution of {ratio} (clipped 5–95th pct)", 380)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Box plots side by side
        fig_box = px.box(
            col_data.assign(**{ratio: clipped}),
            x="Status", y=ratio, color="Status",
            color_discrete_map={"Healthy": HEALTHY_COL, "Distressed": DISTRESS_COL},
            template="plotly_dark", points="outliers",
        )
        style(fig_box, f"Box Plot: {ratio}", 350)
        st.plotly_chart(fig_box, use_container_width=True)

        # Stats table
        stats = col_data.groupby("Status")[ratio].describe().round(3)
        st.dataframe(stats, use_container_width=True)

    # TAB 2 — Correlation heatmap
    with tab2:
        top_cols = ["Net Profit Margin","Total Debt To Capitalization","Current Ratio",
                    "Interest Coverage","Income Quality","Long Term Debt To Capitalization",
                    "Revenue Per Share","Net Income Per Share","Free Cash Flow Per Share",
                    "Earnings Yield","EV To Sales","Dividend Yield"]
        avail = [c for c in top_cols if c in mdf.columns]
        sel_cols = st.multiselect("Select features for correlation matrix", NUM_COLS, default=avail[:10])

        if len(sel_cols) >= 2:
            corr = mdf[sel_cols].dropna().corr().round(2)
            fig_heat = px.imshow(
                corr, text_auto=True, aspect="auto",
                color_continuous_scale=[[0,C["coral"]], [0.5,C["navy"]], [1,C["teal"]]],
                zmin=-1, zmax=1, template="plotly_dark",
            )
            style(fig_heat, "Pearson Correlation Matrix", 560)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Select at least 2 features.")

    # TAB 3 — Ratio comparison distressed vs healthy (uses raw unscaled data + merged label)
    with tab3:
        st.markdown("#### Median Ratio Comparison: Distressed vs Healthy")
        st.markdown(
            f"<div class='info-box'>Values are taken from the <b>original unscaled dataset</b> "
            f"merged with distress labels. This ensures genuine ratio differences are visible "
            f"regardless of which zone filter is active in the sidebar.</div>",
            unsafe_allow_html=True,
        )

        # These columns exist in the raw dataset (not removed by leakage prevention)
        # and show the clearest separation between distressed and healthy firms
        COMPARE_CATEGORIES = {
            "Profitability": [
                "Return On Assets", "Return On Equity",
                "Net Profit Margin", "EBIT Per Revenue", "Gross Profit Margin",
            ],
            "Leverage": [
                "Debt Ratio", "Debt To Assets", "Debt Equity Ratio",
                "Long Term Debt To Capitalization",
            ],
            "Liquidity": [
                "Current Ratio", "Quick Ratio", "Cash Ratio",
            ],
            "Efficiency": [
                "Asset Turnover", "Inventory Turnover", "Receivables Turnover",
            ],
            "Coverage": [
                "Interest Coverage", "Cash Flow Coverage Ratios",
            ],
        }

        # Flatten to a single ordered list, keeping only columns present in raw_labeled
        all_compare = [
            col
            for cols in COMPARE_CATEGORIES.values()
            for col in cols
            if col in raw_labeled.columns
        ]

        # Let user pick a category or show all
        cat_choice = st.selectbox(
            "Filter by category",
            ["All"] + list(COMPARE_CATEGORIES.keys()),
        )
        if cat_choice == "All":
            show_ratios = all_compare
        else:
            show_ratios = [c for c in COMPARE_CATEGORIES[cat_choice] if c in raw_labeled.columns]

        # Build summary from raw_labeled (ALWAYS full dataset, no zone filter)
        summary_rows = []
        for col in show_ratios:
            vals = raw_labeled[col].dropna()
            p1, p99 = vals.quantile(0.01), vals.quantile(0.99)
            h_med = raw_labeled[raw_labeled["Financial_Distress"]==0][col].clip(p1, p99).median()
            d_med = raw_labeled[raw_labeled["Financial_Distress"]==1][col].clip(p1, p99).median()
            diff  = round(d_med - h_med, 4)
            pct   = round((d_med - h_med) / abs(h_med) * 100, 1) if h_med != 0 else np.nan
            summary_rows.append({
                "Ratio":              col,
                "Healthy Median":     round(h_med, 4),
                "Distressed Median":  round(d_med, 4),
                "Difference":         diff,
                "% Change":           pct,
            })
        sumdf = pd.DataFrame(summary_rows)

        # Grouped bar chart
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="Healthy",
            x=sumdf["Ratio"], y=sumdf["Healthy Median"],
            marker_color=HEALTHY_COL, opacity=0.85,
            hovertemplate="%{x}<br>Healthy Median: %{y:.4f}<extra></extra>",
        ))
        fig_cmp.add_trace(go.Bar(
            name="Distressed",
            x=sumdf["Ratio"], y=sumdf["Distressed Median"],
            marker_color=DISTRESS_COL, opacity=0.85,
            hovertemplate="%{x}<br>Distressed Median: %{y:.4f}<extra></extra>",
        ))
        style(fig_cmp, "Median Ratio Comparison — Raw Unscaled Values (clipped 1–99th pct)", 440)
        fig_cmp.update_layout(
            barmode="group",
            xaxis_tickangle=-35,
            xaxis=dict(gridcolor=C["border"]),
            yaxis=dict(gridcolor=C["border"]),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # % difference lollipop chart
        sumdf_sorted = sumdf.dropna(subset=["% Change"]).sort_values("% Change")
        bar_colors = [DISTRESS_COL if v > 0 else HEALTHY_COL for v in sumdf_sorted["% Change"]]
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(
            x=sumdf_sorted["% Change"],
            y=sumdf_sorted["Ratio"],
            orientation="h",
            marker_color=bar_colors,
            opacity=0.85,
            hovertemplate="%{y}<br>% Change: %{x:.1f}%<extra></extra>",
        ))
        fig_diff.add_vline(x=0, line_color=C["muted"], line_width=1.5, line_dash="dash")
        style(fig_diff,
              "% Change in Median: Distressed vs Healthy  "
              "(positive = distressed higher, negative = distressed lower)", 400)
        fig_diff.update_layout(xaxis_title="% Difference (Distressed − Healthy) / |Healthy|",
                                xaxis=dict(gridcolor=C["border"]),
                                yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig_diff, use_container_width=True)

        # Summary table
        st.dataframe(
            sumdf.set_index("Ratio").style.background_gradient(
                subset=["% Change"], cmap="RdYlGn_r", vmin=-100, vmax=100
            ),
            use_container_width=True,
        )

    # TAB 4 — Raw data
    with tab4:
        st.markdown(f"Showing **{min(500, len(mdf)):,}** of **{len(mdf):,}** rows")
        show_cols = st.multiselect("Columns to display", mdf.columns.tolist(),
                                    default=["Year","Stock","Altman_Zone","Z_Score","Financial_Distress",
                                             "Net Profit Margin","Total Debt To Capitalization","Current Ratio"])
        disp = mdf[show_cols].head(500)
        st.dataframe(disp, use_container_width=True, height=420)

        csv = mdf.to_csv(index=False).encode()
        st.download_button("⬇️ Download Filtered Data", csv, "nse_filtered.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Z-SCORE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Z-Score Analysis":
    st.markdown("## Revised Altman Z*-Score Analysis")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🗺️ Zone Map", "📈 Score Trends", "🏢 Stock Deep-Dive"])

    with tab1:
        # Z-Score distribution
        z_clip = mdf["Z_Score"].clip(mdf["Z_Score"].quantile(0.01), mdf["Z_Score"].quantile(0.95))
        fig_zdist = px.histogram(
            mdf.assign(Z_clip=z_clip), x="Z_clip", color="Altman_Zone", nbins=60,
            color_discrete_map={"Healthy": HEALTHY_COL, "Distressed": DISTRESS_COL},
            opacity=0.8, barmode="overlay", template="plotly_dark",
            labels={"Z_clip": "Z*-Score (clipped at 95th pct)"},
        )
        fig_zdist.add_vline(x=1.10, line_dash="dash", line_color=DISTRESS_COL, annotation_text="1.10 (Distress threshold)",
                            annotation_font_color=DISTRESS_COL)
        fig_zdist.add_vline(x=2.60, line_dash="dash", line_color=C["gold"],    annotation_text="2.60 (Healthy threshold)",
                            annotation_font_color=C["gold"])
        style(fig_zdist, "Z*-Score Distribution by Zone", 380)
        st.plotly_chart(fig_zdist, use_container_width=True)


    with tab2:
        yr_zone = model.groupby(["Year","Altman_Zone"]).size().reset_index(name="Count")
        fig_yr = px.area(yr_zone, x="Year", y="Count", color="Altman_Zone",
                          color_discrete_map={"Healthy": HEALTHY_COL, "Distressed": DISTRESS_COL},
                          template="plotly_dark")
        style(fig_yr, "Zone Count Trend Over Years", 370)
        st.plotly_chart(fig_yr, use_container_width=True)

        # Median Z-Score by year
        yr_z = model.groupby("Year")["Z_Score"].agg(["median","mean"]).reset_index()
        fig_zyr = go.Figure()
        fig_zyr.add_trace(go.Scatter(x=yr_z["Year"], y=yr_z["median"], name="Median Z*", mode="lines+markers",
                                      line=dict(color=C["blue"], width=2.5), marker=dict(size=8)))
        fig_zyr.add_trace(go.Scatter(x=yr_z["Year"], y=yr_z["mean"], name="Mean Z*", mode="lines+markers",
                                      line=dict(color=C["gold"], width=2, dash="dot"), marker=dict(size=6)))
        fig_zyr.add_hline(y=1.10, line_dash="dash", line_color=DISTRESS_COL, annotation_text="Distress threshold (1.10)")
        fig_zyr.add_hline(y=2.60, line_dash="dash", line_color=C["gold"],    annotation_text="Healthy threshold (2.60)")
        style(fig_zyr, "Median & Mean Z*-Score by Year", 360)
        st.plotly_chart(fig_zyr, use_container_width=True)

    with tab3:
        st.markdown("#### Z*-Score Component Decomposition by Year")
        st.markdown(
            f"<div class='info-box'>Shows how the four Altman Z*-Score components (X1–X4) "
            f"have evolved over the observation period for <b>distressed vs healthy firms</b>. "
            f"This reveals <i>which component drives</i> the overall Z*-Score signal.</div>",
            unsafe_allow_html=True,
        )

        # Rebuild components from raw_labeled so we have unscaled values
        rl = raw_labeled.copy()
        rl["X1"] = (rl["Working Capital"] / rl["Tangible Asset Value"].replace(0, np.nan)).clip(-10, 10)
        rl["X2"] = rl["Return On Assets"].clip(-1, 1)
        rl["X3"] = (rl["EBIT Per Revenue"] * rl["Asset Turnover"]).clip(-1, 1)
        rl["X4"] = (rl["PB Ratio"] * (1 - rl["Debt Ratio"]) / rl["Debt Ratio"].replace(0, np.nan)).clip(-20, 100)
        rl["Status"] = rl["Financial_Distress"].map({0: "Healthy", 1: "Distressed"})

        # Mean of each component per year × status
        comp_yr = (
            rl.groupby(["Year", "Status"])[["X1","X2","X3","X4"]]
            .median()
            .reset_index()
        )
        comp_labels = {
            "X1": "X1 — Working Capital / TA",
            "X2": "X2 — ROA (Retained Earnings proxy)",
            "X3": "X3 — EBIT / TA",
            "X4": "X4 — MktCap / Book Liabilities",
        }
        comp_colors = {
            "X1": C["blue"],
            "X2": C["teal"],
            "X3": C["gold"],
            "X4": C["coral"],
        }

        c1, c2 = st.columns(2)
        for i, (comp, label) in enumerate(comp_labels.items()):
            fig_comp = go.Figure()
            for status, dash, width in [("Healthy", "solid", 2.5), ("Distressed", "dot", 2.5)]:
                sub = comp_yr[comp_yr["Status"] == status]
                color = HEALTHY_COL if status == "Healthy" else DISTRESS_COL
                fig_comp.add_trace(go.Scatter(
                    x=sub["Year"], y=sub[comp],
                    mode="lines+markers", name=status,
                    line=dict(color=color, width=width, dash=dash),
                    marker=dict(size=7, color=color),
                    hovertemplate=f"{status}<br>Year: %{{x}}<br>{comp}: %{{y:.4f}}<extra></extra>",
                ))
            fig_comp.add_hline(y=0, line_color=C["muted"], line_width=1, line_dash="dash")
            style(fig_comp, label, 280)
            fig_comp.update_layout(
                xaxis=dict(tickmode="linear", dtick=2, gridcolor=C["border"]),
                yaxis=dict(gridcolor=C["border"]),
                legend=dict(orientation="h", y=1.12, x=0),
                margin=dict(l=20, r=20, t=60, b=20),
            )
            col_target = c1 if i % 2 == 0 else c2
            col_target.plotly_chart(fig_comp, use_container_width=True)

        # Bottom: violin plot — distribution of each component by status (latest year)
        st.markdown("##### Component Distribution (Latest Year: 2022)")
        latest_rl = rl[rl["Year"] == rl["Year"].max()].copy()
        comp_long = latest_rl.melt(
            id_vars=["Status"],
            value_vars=list(comp_labels.keys()),
            var_name="Component", value_name="Value",
        )
        comp_long["Component"] = comp_long["Component"].map(comp_labels)

        fig_violin = px.violin(
            comp_long, x="Component", y="Value", color="Status",
            color_discrete_map={"Healthy": HEALTHY_COL, "Distressed": DISTRESS_COL},
            box=True, points=False, template="plotly_dark",
        )
        style(fig_violin, "Component Value Distribution: Distressed vs Healthy (2022)", 400)
        fig_violin.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig_violin, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  ML Model Results":
    st.markdown("## Machine Learning Model Results")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


    # ── Hardcoded results from the paper ──────────────────────────────────────
    exp1 = pd.DataFrame({
        "Model":     ["Logistic Regression","Decision Tree","Random Forest","SVM","XGBoost"],
        "Train Acc": [0.88, 0.99, 0.96, 0.92, 0.99],
        "Test Acc":  [0.81, 0.96, 0.90, 0.84, 0.97],
        "Precision": [0.15, 0.54, 0.29, 0.19, 0.56],
        "Recall":    [0.85, 0.84, 0.92, 0.89, 0.93],
        "F1":        [0.26, 0.66, 0.44, 0.32, 0.70],
        "AUC":       [0.88, 0.91, 0.97, 0.90, 0.99],
    })
    exp2 = pd.DataFrame({
        "Model":     ["Logistic Regression","Decision Tree","Random Forest","SVM","XGBoost"],
        "Train Acc": [0.87, 0.99, 0.97, 0.92, 0.99],
        "Test Acc":  [0.81, 0.96, 0.91, 0.84, 0.97],
        "Precision": [0.16, 0.54, 0.30, 0.19, 0.56],
        "Recall":    [0.85, 0.87, 0.90, 0.89, 0.93],
        "F1":        [0.27, 0.67, 0.46, 0.32, 0.70],
        "AUC":       [0.89, 0.92, 0.97, 0.90, 0.99],
    })
    cart = pd.DataFrame({
        "Sample":    list(range(10)),
        "Ratio":     [f"{r}:1" for r in range(1,11)],
        "Normal":    [475,950,1425,1900,2375,2850,3325,3800,4275,4750],
        "Distress":  [475]*10,
        "Train Acc": [0.93,0.92,0.93,0.93,0.93,0.94,0.94,0.94,0.94,0.95],
        "Train F1":  [0.93,0.89,0.87,0.84,0.82,0.81,0.79,0.78,0.76,0.75],
        "Test Acc":  [0.88,0.91,0.93,0.94,0.94,0.94,0.94,0.95,0.95,0.95],
        "Test F1":   [0.40,0.47,0.53,0.55,0.57,0.58,0.58,0.60,0.60,0.60],
        "Test AUC":  [0.94,0.93,0.92,0.91,0.92,0.93,0.92,0.91,0.94,0.94],
    })
    ann_res = {"Test Acc":0.9419,"Precision":0.4608,"Recall":0.7532,"F1":0.5714,"AUC":0.9762,"Train Acc":0.93}

    tab1,tab2,tab3,tab4 = st.tabs(["⚡ Experiment 1","🎯 Experiment 2 (RFECV)","🔁 CART Bootstrap","🧠 ANN"])

    MODELS_COL = {
        "Logistic Regression": C["slate_l"],
        "Decision Tree":       C["gold"],
        "Random Forest":       C["teal"],
        "SVM":                 C["blue_l"],
        "XGBoost":             C["coral"],
    }

    def radar_chart(df, title):
        metrics = ["Test Acc","Precision","Recall","F1","AUC"]
        fig = go.Figure()
        for _, row in df.iterrows():
            vals = [row[m] for m in metrics] + [row[metrics[0]]]
            base_color = MODELS_COL.get(row["Model"], C["blue"])
            
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=metrics+[metrics[0]], name=row["Model"], fill="toself",
                line=dict(color=base_color, width=2),
                fillcolor=hex_to_rgba(base_color, alpha=0.15) # Fixed transparency property
            ))
        fig.update_layout(**PLOT_LAYOUT, polar=dict(
            bgcolor=C["card"],
            radialaxis=dict(visible=True, range=[0,1], gridcolor=C["border"],
                            tickfont=dict(color=C["muted"], size=9)),
            angularaxis=dict(gridcolor=C["border"], tickfont=dict(color=C["text"]))),
            height=420, title=dict(text=title, font=dict(size=13, color=C["text"])))
        return fig

    def bar_metric(df, metric, title):
        fig = px.bar(df, x="Model", y=metric, color="Model",
                      color_discrete_map=MODELS_COL, template="plotly_dark",
                      text_auto=".3f")
        fig.update_traces(textposition="outside")
        style(fig, title, 320)
        return fig

    with tab1:
        st.markdown(f"<div class='info-box'>Trained on <b>SMOTE+ENN resampled</b> training set. Evaluated on the <b>original unmodified test set</b>. GridSearchCV with 5-fold CV and F1 scoring.</div>", unsafe_allow_html=True)
        c1,c2 = st.columns([3,2])
        with c1:
            st.plotly_chart(radar_chart(exp1, "Experiment 1 — Radar Chart"), use_container_width=True)
        with c2:
            st.dataframe(exp1.set_index("Model").round(3), use_container_width=True, height=220)
            st.markdown(f"""<div class='success-box'>
            🥇 <b>XGBoost wins</b>: AUC 0.99 · F1 0.70 · Test Acc 96.85%<br>
            🏅 ANN (standalone): AUC 0.97 · F1 0.57 · Test Acc 94.19%
            </div>""", unsafe_allow_html=True)

        c3,c4 = st.columns(2)
        with c3: st.plotly_chart(bar_metric(exp1,"AUC","AUC by Model — Experiment 1"), use_container_width=True)
        with c4: st.plotly_chart(bar_metric(exp1,"F1","F1-Score by Model — Experiment 1"), use_container_width=True)

    with tab2:
        st.markdown(f"<div class='info-box'>RFECV selects the optimal feature subset per estimator. The <b>union</b> of selected features is used to retrain all models.</div>", unsafe_allow_html=True)
        c1,c2 = st.columns([3,2])
        with c1:
            st.plotly_chart(radar_chart(exp2, "Experiment 2 — RFECV Radar Chart"), use_container_width=True)
        with c2:
            # Exp1 vs Exp2 diff
            diff_df = exp1.set_index("Model")[["F1","AUC","Test Acc"]].copy()
            diff_df2 = exp2.set_index("Model")[["F1","AUC","Test Acc"]].copy()
            diff_df.columns = ["F1 Exp1","AUC Exp1","Acc Exp1"]
            diff_df2.columns = ["F1 Exp2","AUC Exp2","Acc Exp2"]
            cmp = diff_df.join(diff_df2)
            cmp["ΔF1"] = (cmp["F1 Exp2"] - cmp["F1 Exp1"]).round(3)
            st.dataframe(cmp[["F1 Exp1","F1 Exp2","ΔF1","AUC Exp1","AUC Exp2"]], use_container_width=True, height=220)
            st.markdown(f"""<div class='info-box'>
            RFECV reduces features without meaningful accuracy loss, confirming a <b>parsimonious model</b> is viable.
            </div>""", unsafe_allow_html=True)
        c3,c4 = st.columns(2)
        with c3: st.plotly_chart(bar_metric(exp2,"AUC","AUC — Experiment 2 (RFECV)"), use_container_width=True)
        with c4: st.plotly_chart(bar_metric(exp2,"F1","F1-Score — Experiment 2 (RFECV)"), use_container_width=True)

    with tab3:
        st.markdown(f"<div class='info-box'>CART trained on <b>original (non-SMOTE) training data</b> at 10 class imbalance ratios (1:1 to 10:1). Tests model stability across real-world prevalence levels.</div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            fig_cart_f1 = go.Figure()
            fig_cart_f1.add_trace(go.Scatter(x=cart["Ratio"], y=cart["Test F1"],  name="Test F1",  mode="lines+markers",
                                             line=dict(color=C["teal"],  width=2.5), marker=dict(size=9)))
            fig_cart_f1.add_trace(go.Scatter(x=cart["Ratio"], y=cart["Train F1"], name="Train F1", mode="lines+markers",
                                             line=dict(color=C["blue_l"], width=2, dash="dot"), marker=dict(size=7)))
            style(fig_cart_f1, "CART — F1 by Imbalance Ratio", 320)
            st.plotly_chart(fig_cart_f1, use_container_width=True)
        with c2:
            fig_cart_auc = go.Figure()
            fig_cart_auc.add_trace(go.Scatter(x=cart["Ratio"], y=cart["Test AUC"], name="Test AUC", mode="lines+markers",
                                             line=dict(color=C["coral"], width=2.5), marker=dict(size=9)))
            fig_cart_auc.add_trace(go.Scatter(x=cart["Ratio"], y=cart["Test Acc"], name="Test Acc", mode="lines+markers",
                                             line=dict(color=C["gold"],  width=2, dash="dot"), marker=dict(size=7)))
            style(fig_cart_auc, "CART — AUC & Accuracy by Imbalance Ratio", 320)
            st.plotly_chart(fig_cart_auc, use_container_width=True)

        st.dataframe(cart.set_index("Sample"), use_container_width=True, height=220)
        st.markdown(f"""<div class='success-box'>
        Best Test F1 (0.60) and Test AUC (0.94) achieved at <b>10:1 ratio</b> — performance improves monotonically as
        the model sees more Normal examples, confirming robustness under real-world class distributions.
        </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown(f"<div class='info-box'>Feedforward MLP: <b>3 hidden layers (64→32→16)</b> · ReLU · Adam · batch=35 · max 100 epochs · early stopping (patience=10)</div>", unsafe_allow_html=True)

        a1,a2,a3,a4,a5 = st.columns(5)
        a1.metric("Test Accuracy", f"{ann_res['Test Acc']*100:.2f}%")
        a2.metric("AUC",           f"{ann_res['AUC']:.4f}")
        a3.metric("F1-Score",      f"{ann_res['F1']:.4f}")
        a4.metric("Precision",     f"{ann_res['Precision']:.4f}")
        a5.metric("Recall",        f"{ann_res['Recall']:.4f}")

        # Comparison bar
        all_models_auc = pd.DataFrame({
            "Model": ["Logistic Reg.","SVM","Random Forest","ANN","Decision Tree","XGBoost"],
            "AUC":   [0.88, 0.90, 0.97, 0.9762, 0.91, 0.99],
            "F1":    [0.26, 0.32, 0.44, 0.5714, 0.66, 0.70],
        }).sort_values("AUC")
        COLORS_ALL = [C["slate_l"],C["slate"],C["teal"],C["blue"],C["gold"],C["coral"]]
        fig_all_auc = px.bar(all_models_auc, x="AUC", y="Model", orientation="h",
                              color="Model", color_discrete_sequence=COLORS_ALL,
                              text_auto=".4f", template="plotly_dark")
        style(fig_all_auc, "AUC Comparison — All Models", 340)
        fig_all_auc.update_traces(textposition="outside")
        st.plotly_chart(fig_all_auc, use_container_width=True)

        st.markdown(f"""<div class='success-box'>
        The ANN achieves near-perfect AUC (0.9762) with <b>no overfitting</b> — training and test accuracy within 1% of each other.
        This replicates the reference paper's (Lokanan &amp; Ramzan 2024) finding that neural networks are highly robust for
        financial distress prediction.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SURVIVAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⏳  Survival Analysis":
    st.markdown("## Survival Analysis — Time-to-Financial Distress")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉 Kaplan-Meier","📊 Quartile Analysis","🔬 Cox PH Model"])

    with tab1:
        st.markdown(f"""
        <div class='info-box'>
        <b>1,604 firms</b> started in non-distress · <b>153 distress events</b> (9.5%) · <b>1,451 censored</b><br>
        Duration = years from first observation to first Z* &lt; 1.10 crossing (or end of study).
        </div>""", unsafe_allow_html=True)

        # Compute KM manually from TTE
        tte_sorted = tte.sort_values("Duration")
        times = sorted(tte_sorted[tte_sorted["Event"]==1]["Duration"].unique())
        n_risk = len(tte_sorted)
        surv = 1.0
        km_rows = [{"Time":0, "Survival":1.0}]
        for t in times:
            d = (tte_sorted[(tte_sorted["Duration"]==t) & (tte_sorted["Event"]==1)]).shape[0]
            surv = surv * (1 - d/n_risk)
            n_risk -= (tte_sorted[tte_sorted["Duration"]==t]).shape[0]
            km_rows.append({"Time":t, "Survival":round(surv,4)})
        km_df = pd.DataFrame(km_rows)

        fig_km = go.Figure()
        fig_km.add_trace(go.Scatter(
        x=km_df["Time"], y=km_df["Survival"], mode="lines",
        name="All Firms", line=dict(color=C["blue"], width=3, shape="hv"),
        fill="tozeroy", 
        fillcolor=hex_to_rgba(C["blue"], alpha=0.10),  # <--- Changed this line
        ))
        fig_km.add_hline(y=0.5, line_dash="dash", line_color=C["muted"], annotation_text="Median threshold (50%)")
        # 1. Apply the base layout and general chart settings
        fig_km.update_layout(
            **PLOT_LAYOUT, 
            height=420,
            title=dict(text="Kaplan-Meier: Overall Survival Probability — NSE Firms (2012–2022)", font=dict(size=13))
        )
        
        # 2. Update specific axis settings separately to avoid conflicts
        fig_km.update_yaxes(range=[0, 1.05], gridcolor=C["border"])
        fig_km.update_xaxes(gridcolor=C["border"])
        fig_km.add_annotation(x=9, y=0.93,
                               text=f"n=1,604 firms<br>153 events<br>1,451 censored",
                               showarrow=False, bgcolor=C["card2"], bordercolor=C["border"],
                               font=dict(color=C["text"], size=11), align="left")
        st.plotly_chart(fig_km, use_container_width=True)

        # Survival table
        st.dataframe(km_df.rename(columns={"Time":"Year","Survival":"P(Non-Distressed)"}).set_index("Year"), use_container_width=True, height=180)

    with tab2:
        # KM by quartile
        tte_q = tte[tte["Z_Score"].notna()].copy()
        tte_q["Quartile"] = pd.qcut(tte_q["Z_Score"], q=4, labels=["Q1 (Highest Risk)","Q2","Q3","Q4 (Lowest Risk)"])
        QCOLS = {
            "Q1 (Highest Risk)": C["coral"],
            "Q2":                C["gold"],
            "Q3":                C["teal"],
            "Q4 (Lowest Risk)":  C["blue"],
        }

        fig_qkm = go.Figure()
        for q in ["Q1 (Highest Risk)","Q2","Q3","Q4 (Lowest Risk)"]:
            sub = tte_q[tte_q["Quartile"]==q].sort_values("Duration")
            t_list = sorted(sub[sub["Event"]==1]["Duration"].unique())
            nr = len(sub); s = 1.0
            rows = [{"Time":0,"Survival":1.0}]
            for t in t_list:
                d = sub[(sub["Duration"]==t) & (sub["Event"]==1)].shape[0]
                s = s * (1 - d/nr)
                nr -= sub[sub["Duration"]==t].shape[0]
                rows.append({"Time":t,"Survival":round(s,4)})
            kdf = pd.DataFrame(rows)
            fig_qkm.add_trace(go.Scatter(
                x=kdf["Time"], y=kdf["Survival"], mode="lines", name=q,
                line=dict(color=QCOLS[q], width=2.5, shape="hv"),
            ))
        # 1. Apply the base layout and title
        fig_qkm.update_layout(
            **PLOT_LAYOUT, 
            height=420,
            title=dict(text="KM Curves by Revised Altman Z*-Score Quartile", font=dict(size=13))
        )
        
        # 2. Safely apply specific X and Y axis settings (including titles)
        fig_qkm.update_xaxes(
            title_text="Years since first observation"
        )
        fig_qkm.update_yaxes(
            title_text="P(remaining non-distressed)", 
            range=[0, 1.05], 
            gridcolor=C["border"]
        )

        # Quartile event rates bar
        q_rates = []
        for q in ["Q1 (Highest Risk)","Q2","Q3","Q4 (Lowest Risk)"]:
            sub = tte_q[tte_q["Quartile"]==q]
            q_rates.append({"Quartile": q, "Event Rate %": round(sub["Event"].mean()*100,1), "Firms": len(sub), "Events": sub["Event"].sum()})
        qr_df = pd.DataFrame(q_rates)

        c1,c2 = st.columns([2,1])
        with c1:
            fig_qrate = px.bar(qr_df, x="Quartile", y="Event Rate %", color="Quartile",
                                color_discrete_map={k:v for k,v in QCOLS.items()},
                                text_auto=".1f", template="plotly_dark")
            fig_qrate.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
            style(fig_qrate, "Distress Event Rate by Quartile (Log-rank p < 0.001 ***)", 340)
            st.plotly_chart(fig_qrate, use_container_width=True)
        with c2:
            st.dataframe(qr_df.set_index("Quartile"), use_container_width=True, height=180)
            st.markdown(f"""<div class='warn-box'>
            Q1 firms face a <b>{qr_df.iloc[0]['Event Rate %']:.1f}%</b> distress rate —
            <b>{qr_df.iloc[0]['Event Rate %']/qr_df.iloc[3]['Event Rate %']:.0f}×</b>
            higher than Q4 firms ({qr_df.iloc[3]['Event Rate %']:.1f}%)
            </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown(f"""
        <div class='info-box'>
        Cox PH model fitted with <b>penaliser=0.1</b> (L2 regularisation). Baseline-year financial ratios used as
        time-fixed covariates. <b>C-index = 0.706</b> — model correctly ranks 70.6% of firm pairs by time-to-distress.
        </div>""", unsafe_allow_html=True)

        cox_df = pd.DataFrame({
            "Feature":      ["Total Debt To Capitalization","Income Quality","Interest Coverage",
                             "Current Ratio","Net Profit Margin"],
            "Category":     ["Leverage","Accruals","Debt Service","Liquidity","Profitability"],
            "Beta":         [0.856,  0.006, -0.000, -0.019, -0.783],
            "Hazard Ratio": [2.354,  1.006,  1.000,  0.982,  0.457],
            "SE":           [0.176,  0.007,  0.000,  0.013,  0.239],
            "z-stat":       [4.869,  0.907, -0.582, -1.466, -3.280],
            "p-value":      [0.000,  0.364,  0.561,  0.143,  0.001],
            "HR_lo":        [1.668,  0.993,  0.999,  0.958,  0.286],
            "HR_hi":        [3.323,  1.019,  1.000,  1.006,  0.730],
            "Sig":          ["***", "", "", "", "**"],
        })

        # Forest plot
        colors_forest = [DISTRESS_COL if hr > 1 else HEALTHY_COL for hr in cox_df["Hazard Ratio"]]
        fig_forest = go.Figure()
        for i, row in cox_df.iterrows():
            fig_forest.add_trace(go.Scatter(
                x=[row["HR_lo"], row["Hazard Ratio"], row["HR_hi"]],
                y=[row["Feature"]]*3, mode="lines",
                line=dict(color=colors_forest[i], width=2),
                showlegend=False,
            ))
            fig_forest.add_trace(go.Scatter(
                x=[row["Hazard Ratio"]], y=[row["Feature"]], mode="markers+text",
                marker=dict(size=14, color=colors_forest[i], symbol="diamond"),
                text=[f"  HR={row['Hazard Ratio']:.3f} {row['Sig']}"],
                textposition="middle right", textfont=dict(color=C["text"], size=11),
                showlegend=False,
            ))
        fig_forest.add_vline(x=1.0, line_dash="dash", line_color=C["muted"], line_width=1.5)
        # 1. Apply the base layout and title
        fig_forest.update_layout(
            **PLOT_LAYOUT, 
            height=380,
            title=dict(text="Cox PH Model — Hazard Ratios with 95% CI  |  C-index = 0.706", font=dict(size=13))
        )
        
        # 2. Update the X-axis (including the title and range)
        fig_forest.update_xaxes(
            title_text="Hazard Ratio (HR = 1 : no effect)",
            range=[0, 4], 
            gridcolor=C["border"]
        )
        
        # 3. Update the Y-axis
        fig_forest.update_yaxes(
            gridcolor=C["border"]
        )

        c1,c2 = st.columns([3,2])
        with c1:
            st.dataframe(cox_df[["Feature","Category","Hazard Ratio","p-value","Sig","HR_lo","HR_hi"]].set_index("Feature").round(3),
                         use_container_width=True, height=200)
        with c2:
            st.markdown(f"""
            <div class='warn-box'>
            <b>Total Debt to Capitalisation</b><br>
            HR = 2.354 (p &lt; 0.001 ***)<br>
            Each unit ↑ raises distress hazard by <b>135%</b>
            </div>
            <div class='success-box' style='margin-top:10px;'>
            <b>Net Profit Margin</b><br>
            HR = 0.457 (p = 0.001 **)<br>
            Each unit ↑ reduces distress hazard by <b>54%</b>
            </div>""", unsafe_allow_html=True)

        # HR bar chart
        fig_hr_bar = px.bar(
            cox_df.sort_values("Hazard Ratio"), x="Hazard Ratio", y="Feature", orientation="h",
            color="Category", template="plotly_dark",
            color_discrete_sequence=[C["coral"],C["blue"],C["gold"],C["teal"],C["slate_l"]],
            text="Hazard Ratio",
        )
        fig_hr_bar.add_vline(x=1.0, line_dash="dash", line_color=C["muted"])
        fig_hr_bar.update_traces(texttemplate="%{x:.3f}", textposition="outside")
        style(fig_hr_bar, "Hazard Ratios by Feature", 320)
        st.plotly_chart(fig_hr_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FIRM RISK LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Firm Risk Lookup":
    st.markdown("## Firm Risk Lookup & Z*-Score Calculator")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🏢 Stock History Lookup", "🧮 Z*-Score Calculator"])

    with tab1:
        sel = st.selectbox("Select a stock", all_stocks)
        sdata = model[model["Stock"]==sel].sort_values("Year")

        if len(sdata) == 0:
            st.warning("Stock not found in model-ready dataset.")
        else:
            latest = sdata.iloc[-1]
            zone   = latest["Altman_Zone"]
            z_val  = latest["Z_Score"]

            badge_html = f'<span class="badge-distressed">DISTRESSED</span>' if zone=="Distressed" else f'<span class="badge-healthy">HEALTHY</span>'
            st.markdown(f"""
            <div style='background:{C["card"]};border:1px solid {C["border"]};border-radius:14px;padding:20px 24px;margin-bottom:16px;'>
              <div style='font-size:1.5rem;font-weight:700;'>{sel}</div>
              <div style='margin-top:8px;'>
                Status (latest year): {badge_html} &nbsp;
                Z*-Score: <b style='color:{DISTRESS_COL if zone=="Distressed" else HEALTHY_COL};font-size:1.1rem;'>{z_val:.2f}</b>
              </div>
            </div>""", unsafe_allow_html=True)

            c1,c2,c3 = st.columns(3)
            c1.metric("Latest Year",   str(int(latest["Year"])))
            c2.metric("Z*-Score",      f"{z_val:.2f}")
            c3.metric("Years Observed",str(len(sdata)))

            # Trajectory
            fig_traj = go.Figure()
            zone_col_map = sdata["Altman_Zone"].map({"Healthy":HEALTHY_COL,"Distressed":DISTRESS_COL})
            fig_traj.add_hrect(y0=-100, y1=1.10, fillcolor=DISTRESS_COL, opacity=0.06, layer="below")
            fig_traj.add_hrect(y0=2.60, y1=5000,  fillcolor=HEALTHY_COL,  opacity=0.04, layer="below")
            z_plot = sdata["Z_Score"].clip(-100, sdata["Z_Score"].quantile(0.98) if len(sdata)>2 else 5000)
            fig_traj.add_trace(go.Scatter(
                x=sdata["Year"], y=z_plot, mode="lines+markers",
                line=dict(color=C["blue_l"], width=3),
                marker=dict(size=11, color=zone_col_map.values, line=dict(width=2, color=C["text"])),
                hovertemplate="Year: %{x}<br>Z*: %{y:.2f}<extra></extra>",
            ))
            fig_traj.add_hline(y=1.10, line_dash="dash", line_color=DISTRESS_COL, annotation_text="Distress (1.10)")
            fig_traj.add_hline(y=2.60, line_dash="dash", line_color=C["gold"],    annotation_text="Healthy (2.60)")
            style(fig_traj, f"{sel} — Z*-Score Trajectory", 380)
            fig_traj.update_layout(xaxis=dict(tickmode="linear", dtick=1, gridcolor=C["border"]))
            st.plotly_chart(fig_traj, use_container_width=True)

            # Radar: key ratios vs sector median
            key_r = ["Net Profit Margin","Total Debt To Capitalization","Current Ratio","Interest Coverage","Income Quality"]
            avail_r = [r for r in key_r if r in sdata.columns]
            latest_vals = [latest.get(r, np.nan) for r in avail_r]
            sector_meds = [model[model["Financial_Distress"]==0][r].median() for r in avail_r]

            if any(~np.isnan(v) for v in latest_vals):
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=latest_vals+[latest_vals[0]], theta=avail_r+[avail_r[0]],name=sel, fill="toself", line=dict(color=C["blue"], width=2), fillcolor=hex_to_rgba(C["blue"], alpha=0.20)))
                # 1. Add the trace using the rgba helper to fix the transparency error
                fig_radar.add_trace(go.Scatterpolar(
                    r=sector_meds+[sector_meds[0]], 
                    theta=avail_r+[avail_r[0]],
                    name="Healthy Median", 
                    fill="toself", 
                    line=dict(color=HEALTHY_COL, width=2, dash="dot"), 
                    fillcolor=hex_to_rgba(HEALTHY_COL, alpha=0.15)  # <--- Hex fix applied here
                ))
                fig_radar.update_layout(
                    **PLOT_LAYOUT, 
                    height=360, 
                    title=dict(text=f"{sel} vs Healthy Median (latest year)", font=dict(size=12))
                )
                fig_radar.update_polars(
                    bgcolor=C["card"],
                    radialaxis=dict(visible=True, gridcolor=C["border"], tickfont=dict(color=C["muted"], size=9)),
                    angularaxis=dict(gridcolor=C["border"], tickfont=dict(color=C["text"]))
                )

                st.plotly_chart(fig_radar, use_container_width=True)

            # History table
            disp_cols = ["Year","Altman_Zone","Z_Score"] + avail_r
            st.dataframe(sdata[disp_cols].set_index("Year").round(3), use_container_width=True)

    with tab2:
        st.markdown("#### Manual Z*-Score Calculator")
        st.markdown(f"<div class='info-box'>Enter the four Z*-Score inputs for any firm to instantly compute its Revised Altman Z*-Score and risk zone.</div>", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            x1 = st.number_input("X₁ = Working Capital / Total Assets",           min_value=-10.0, max_value=10.0,  value=0.25, step=0.01, format="%.4f")
            x2 = st.number_input("X₂ = Return on Assets (ROA proxy)",              min_value=-5.0,  max_value=5.0,   value=0.05, step=0.01, format="%.4f")
        with c2:
            x3 = st.number_input("X₃ = EBIT / Total Assets",                        min_value=-2.0,  max_value=2.0,   value=0.08, step=0.01, format="%.4f")
            x4 = st.number_input("X₄ = Market Value Equity / Book Value Liabilities",min_value=-10.0, max_value=500.0, value=5.0,  step=0.1,  format="%.3f")

        z_calc = 3.25 + 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4

        if z_calc < 1.10:
            zone_c, zone_t, zone_col = C["card"], "🔴  DISTRESSED ZONE  (Z* < 1.10)", DISTRESS_COL
        elif z_calc <= 2.60:
            zone_c, zone_t, zone_col = C["card"], "🟡  GREY ZONE  (1.10 ≤ Z* ≤ 2.60)", C["gold"]
        else:
            zone_c, zone_t, zone_col = C["card"], "🟢  HEALTHY ZONE  (Z* > 2.60)", HEALTHY_COL

        st.markdown(f"""
        <div style='background:{zone_c};border:2px solid {zone_col};border-radius:14px;padding:24px 28px;text-align:center;margin-top:16px;'>
          <div style='font-size:3rem;font-weight:800;color:{zone_col};'>{z_calc:.4f}</div>
          <div style='font-size:1.1rem;font-weight:600;color:{zone_col};margin-top:8px;'>{zone_t}</div>
          <div style='font-size:0.85rem;color:{C["muted"]};margin-top:8px;'>
            Z* = 3.25 + 6.56×{x1} + 3.26×{x2} + 6.72×{x3} + 1.05×{x4}
          </div>
        </div>""", unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(max(z_calc, -5), 30),
            number=dict(font=dict(color=C["text"], size=32)),
            gauge=dict(
                axis=dict(range=[-5, 30], tickcolor=C["muted"], tickfont=dict(color=C["muted"])),
                bar=dict(color=zone_col, thickness=0.35),
                bgcolor=C["card2"],
                steps=[
                    dict(range=[0, 1.10], color=hex_to_rgba(C["red"], alpha=0.20)), 
                    dict(range=[1.10, 2.60], color=hex_to_rgba(C["gold"], alpha=0.20)),
                    dict(range=[2.60, 5.0], color=hex_to_rgba(C["teal"], alpha=0.20))
                ],
                threshold=dict(line=dict(color=zone_col, width=4), thickness=0.75, value=z_calc),
            ),
            title=dict(text="Z*-Score Gauge", font=dict(color=C["text"], size=14)),
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"]), height=280, margin=dict(l=30,r=30,t=40,b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
