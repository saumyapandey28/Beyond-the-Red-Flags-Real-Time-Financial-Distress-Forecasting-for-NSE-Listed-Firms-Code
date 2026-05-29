"""
Survival Analysis — Financial Distress Duration
=================================================
INPUT  : nse_altman_corrected.csv
         (output of altman_z_score_corrected.py — ALL original feature
          columns intact, plus Z_Score, Altman_Zone, Financial_Distress)

OUTPUT : survival_km_overall.png        (overall Kaplan-Meier curve)
         survival_km_zscore_groups.png  (KM curves by Z*-Score quartile)
         survival_km_logrank.png        (KM + log-rank test panel)
         survival_cox_results.csv       (Cox PH coefficients + hazard ratios)
         survival_cox_forest_plot.png   (hazard ratio forest plot)
         survival_time_to_distress.csv  (per-firm time-to-event data)

METHODS: Kaplan-Meier Estimator + Cox Proportional Hazards Model

REQUIREMENT: pip install lifelines matplotlib

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW COX FEATURES WERE SELECTED — FULL JUSTIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cox PH regression has three hard constraints that classification
models (LR, RF, SVM, ANN) do not share:

CONSTRAINT 1 — EVENTS-PER-VARIABLE (EPV) RULE
  The Cox model becomes statistically unreliable when there are too
  few observed events (distress occurrences) relative to the number
  of covariates. Peduzzi et al. (1995), the foundational reference
  on this, set the minimum at EPV >= 10 (i.e., at least 10 distress
  events per covariate in the model).

  In this dataset, ~1,400+ distress events are observed across the
  TTE table (all firms starting as non-distressed), giving an upper
  ceiling of 1,400 / 10 = 140 safe variables. This is well above
  what we need, so EPV is not the binding constraint here — but it
  IS the reason we do not simply dump all 73 available features into
  Cox: many would be statistically redundant or collinear.

CONSTRAINT 2 — NO CORRELATED FEATURES
  Unlike tree-based models (RF, GBM) that handle correlated features
  by feature splitting, Cox PH inflates standard errors when predictors
  are correlated. This makes individual hazard ratios unreliable and
  hard to interpret. For example:
    - Current Ratio and Quick Ratio have r = 0.668 — including both
      would give unstable HRs for both, with neither interpretable.
    - Free Cash Flow Per Share and Dividend Yield have r = 0.920 —
      nearly identical signals; including both is statistically wrong.
  Rule applied: only one feature per highly correlated cluster.

CONSTRAINT 3 — ONE REPRESENTATIVE PER THEORETICAL DIMENSION
  Cox is a causal-interpretive model — we want to understand WHICH
  financial health dimension drives faster distress onset. If two
  features measure the same underlying construct (e.g., two leverage
  ratios), their individual HRs split the same effect and neither
  can be cleanly interpreted. The standard practice in survival
  literature is to select one ratio per theoretical category.

FEATURE SELECTION PROCESS (applied to the 73 available features):

  Step 1 — Remove Z-Score inputs
    25 columns removed (they created the label — cannot be features).
    73 columns remain available.

  Step 2 — Remove duplicates and near-identical columns
    E.g., 'Payout Ratio' = 'Dividend Payout Ratio' (same metric,
    different name). 'Days Of Sales Outstanding' = 'Days Sales
    Outstanding'. These are removed, leaving distinct concepts only.

  Step 3 — Group remaining features by financial theory dimension
    Each group covers one aspect of financial health:
      Liquidity     : how easily can the firm meet short-term obligations?
      Profitability : is the firm generating enough income?
      Leverage      : how much debt is the firm carrying?
      Debt service  : can the firm actually service its debt payments?
      Accruals      : is the reported income backed by real cash flows?

  Step 4 — Choose the best representative from each group
    Criterion: well-established in financial distress literature,
    lowest pairwise correlation with other selected features,
    lowest missing data rate.

FINAL 5 FEATURES — ONE PER DIMENSION:

  1. Current Ratio        — LIQUIDITY
     Why chosen: most universally used single liquidity indicator across
     Altman (1968), Ohlson (1980), and subsequent distress literature.
     Why not Quick Ratio: correlated with Current Ratio (r=0.668);
     Current Ratio is more commonly cited in Indian market research.

  2. Net Profit Margin    — PROFITABILITY
     Why chosen: captures bottom-line profitability after all expenses,
     including financing costs. More sensitive to distress than Gross
     Profit Margin because it includes interest and tax effects.
     Why not Gross Profit Margin: GPM misses leverage costs, which are
     a primary driver of financial distress in Indian firms.

  3. Interest Coverage    — DEBT SERVICE CAPACITY
     Why chosen: directly measures whether operating profits cover
     interest payments. When Interest Coverage < 1, the firm cannot
     service debt from operations — a classic leading indicator of
     distress onset. Widely used in credit risk and survival literature.

  4. Total Debt To Capitalization  — LEVERAGE
     Why chosen: measures total debt as a proportion of total
     capitalisation (debt + equity), giving a clean balance-sheet
     snapshot of capital structure risk. Preferred over Long Term Debt
     To Capitalization because it captures short-term debt too, which
     is common in Indian firms.

  5. Income Quality       — ACCRUALS / CASH FLOW QUALITY
     Why chosen: defined as Operating Cash Flow / Net Income. When this
     ratio is below 1, reported earnings exceed actual cash generation —
     a signal of earnings manipulation or unsustainable accruals. Lower
     Income Quality is associated with faster deterioration to distress.
     This is the only cash-flow-based feature, capturing a dimension
     that pure ratio measures miss.

WHAT THIS GIVES YOU:
  Five features, zero pairwise correlation above r=0.35, covering
  five distinct theoretical dimensions of financial health. Each
  hazard ratio from Cox is independently interpretable:
    HR for Current Ratio   → does holding more liquid assets delay distress?
    HR for Net Profit Margin → does profitability buffer delay distress onset?
    HR for Interest Coverage → does debt-service headroom extend survival?
    HR for Leverage          → does higher debt load accelerate distress?
    HR for Income Quality    → do accrual-heavy firms fail sooner?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
except ImportError:
    raise ImportError("Run: pip install lifelines")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: LOADING DATA")
print("=" * 65)

# MUST use nse_altman_corrected.csv (pre-feature-removal file).
# nse_altman_model_ready.csv has leakage columns stripped — that file
# is missing the features needed for the Cox model.
df = pd.read_csv('nse_altman_corrected.csv', low_memory=False)
df['Year']               = df['Year'].astype(int)
df['Financial_Distress'] = df['Financial_Distress'].astype(int)
df.sort_values(['Stock', 'Year'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  Rows   : {len(df):,}")
print(f"  Stocks : {df['Stock'].nunique():,}")
print(f"  Years  : {df['Year'].min()} - {df['Year'].max()}")
print(f"  Distressed firm-years : {df['Financial_Distress'].sum():,} "
      f"({df['Financial_Distress'].mean()*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: BUILD TIME-TO-EVENT TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: BUILDING TIME-TO-EVENT TABLE")
print("=" * 65)

# 5 Cox features — one per financial health dimension
# Each chosen for theoretical relevance, low mutual correlation,
# and independence from Z-Score components.
# See module docstring for full selection justification.
COX_FEATURES = [
    'Current Ratio',                  # Liquidity
    'Net Profit Margin',              # Profitability
    'Interest Coverage',              # Debt service capacity
    'Total Debt To Capitalization',   # Leverage
    'Income Quality',                 # Accruals / cash flow quality
]
# Filter to columns actually present in this file
COX_FEATURES = [c for c in COX_FEATURES if c in df.columns]

records = []
for stock, grp in df.groupby('Stock'):
    grp = grp.sort_values('Year').reset_index(drop=True)

    # Exclude firms already distressed at first observation
    # (no "time to distress" to measure)
    if grp['Financial_Distress'].iloc[0] == 1:
        continue

    first_year = grp['Year'].iloc[0]
    last_year  = grp['Year'].iloc[-1]

    # First distress event (if any)
    distress_rows = grp[grp['Financial_Distress'] == 1]
    if len(distress_rows) > 0:
        event_year = distress_rows['Year'].iloc[0]
        duration   = max(event_year - first_year, 1)
        event      = 1
    else:
        # Censored: observed until last year without distress
        duration   = max(last_year - first_year, 1)
        event      = 0

    row = {
        'Stock'   : stock,
        'Duration': duration,
        'Event'   : event,
        'Z_Score' : grp['Z_Score'].iloc[0],   # baseline Z* for quartile grouping
    }
    for col in COX_FEATURES:
        row[col] = grp[col].iloc[0]            # baseline (first-year) value

    records.append(row)

tte = pd.DataFrame(records)

n_events = tte['Event'].sum()
n_feats  = len(COX_FEATURES)
epv      = n_events / n_feats

print(f"  Firms included (started non-distressed): {len(tte):,}")
print(f"  Distress events  (event=1)             : {n_events:,} "
      f"({tte['Event'].mean()*100:.1f}%)")
print(f"  Censored         (event=0)             : {(tte['Event']==0).sum():,}")
print(f"  Duration range                         : "
      f"{tte['Duration'].min()}-{tte['Duration'].max()} years")
print(f"\n  Cox model EPV check:")
print(f"    Events = {n_events:,}  /  Features = {n_feats}  =  EPV = {epv:.0f}")
print(f"    (Minimum recommended EPV = 10; our EPV = {epv:.0f} -- well above threshold)")

tte.to_csv('survival_time_to_distress.csv', index=False)
print(f"\n  Saved -> 'survival_time_to_distress.csv'")

# Pairwise correlation check on Cox features
print(f"\n  Pairwise correlations in Cox feature set (should all be < 0.70):")
cox_corr = tte[COX_FEATURES].dropna().corr().abs()
upper = cox_corr.where(np.triu(np.ones(cox_corr.shape), k=1).astype(bool))
any_high = False
for c1 in upper.index:
    for c2 in upper.columns:
        v = upper.loc[c1, c2]
        if not pd.isna(v):
            flag = " <-- HIGH" if v >= 0.50 else ""
            print(f"    {c1} vs {c2}: r={v:.3f}{flag}")
            if v >= 0.70:
                any_high = True
if not any_high:
    print("    All pairs below r=0.70 -- no multicollinearity concern.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Z*-SCORE QUARTILE GROUPING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Z*-SCORE QUARTILE GROUPING")
print("=" * 65)

tte_clean = tte.dropna(subset=['Z_Score']).copy()

QUARTILE_LABELS = {
    1: 'Q1 (highest risk)',
    2: 'Q2',
    3: 'Q3',
    4: 'Q4 (lowest risk)'
}
tte_clean['Z_Quartile'] = pd.qcut(
    tte_clean['Z_Score'], q=4, labels=[1, 2, 3, 4]
).astype(int)
tte_clean['Z_Group'] = tte_clean['Z_Quartile'].map(QUARTILE_LABELS)

print(f"\n  Z*-Score quartile breakdown:")
for q in [1, 2, 3, 4]:
    mask = tte_clean['Z_Quartile'] == q
    sub  = tte_clean[mask]
    n_ev = sub['Event'].sum()
    print(f"  {QUARTILE_LABELS[q]:20s}  n={len(sub):4d}  "
          f"Z* [{sub['Z_Score'].min():.2f}, {sub['Z_Score'].max():.2f}]  "
          f"events={n_ev:3d}  rate={n_ev/len(sub)*100:.1f}%")
print(f"\n  Expectation: Q1 event rate > Q2 > Q3 > Q4 confirms Z* stratifies by risk.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: KAPLAN-MEIER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: KAPLAN-MEIER ESTIMATOR")
print("=" * 65)

PALETTE = {
    'Q1 (highest risk)': '#D85A30',
    'Q2'               : '#BA7517',
    'Q3'               : '#1D9E75',
    'Q4 (lowest risk)' : '#185FA5',
}

# ── Overall KM ──────────────────────────────────────────────────────────────
kmf_overall = KaplanMeierFitter()
kmf_overall.fit(tte['Duration'], event_observed=tte['Event'], label='All firms')
print(f"  Overall median survival : {kmf_overall.median_survival_time_} years")
print(f"  (inf = more than 50% of firms never became distressed in the study window)")
# -------------------------------------------------------------------------
# NEW: PRINT YEARLY SURVIVAL PROBABILITIES
# -------------------------------------------------------------------------
print("\n  Overall Probability of Remaining Healthy (Survival) by Year:")
print("  " + "-" * 45)
print("  Year | Survival Probability | Total Bankruptcies")
print("  " + "-" * 45)

max_duration = int(tte['Duration'].max())
for year in range(1, max_duration + 1):
    # lifelines built-in function to get exact probability at time t
    prob = kmf_overall.predict(year)
    
    # Calculate cumulative events up to this year
    events_to_date = tte[(tte['Duration'] <= year) & (tte['Event'] == 1)].shape[0]
    
    print(f"    {year:2d} |        {prob:6.2f}       |        {events_to_date:3d}")
print("  " + "-" * 45 + "\n")
# -------------------------------------------------------------------------
fig_o, ax_o = plt.subplots(figsize=(8, 5))
kmf_overall.plot_survival_function(
    ax=ax_o, ci_show=True, color='#185FA5', linewidth=2.0
)
ax_o.set_title('Kaplan-Meier: Overall Survival Probability\n'
               'NSE-Listed Firms (2012-2022)', fontsize=12)
ax_o.set_xlabel('Years since first observation', fontsize=11)
ax_o.set_ylabel('P(remaining non-distressed)', fontsize=11)
ax_o.set_ylim(0, 1.05)
ax_o.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
ax_o.text(0.3, 0.52, 'Median survival threshold (50%)',
          fontsize=8, color='grey', transform=ax_o.get_yaxis_transform())
ax_o.grid(True, alpha=0.3)
ax_o.annotate(
    f"n={len(tte):,} firms\n{tte['Event'].sum()} distress events\n"
    f"{(tte['Event']==0).sum()} censored",
    xy=(0.98, 0.98), xycoords='axes fraction',
    ha='right', va='top', fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
)
plt.tight_layout()
plt.savefig('survival_km_overall.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved -> 'survival_km_overall.png'")

# ── KM by Z*-Score quartile ─────────────────────────────────────────────────
fig_q, ax_q = plt.subplots(figsize=(9, 6))
kmf_list = {}

for q in [1, 2, 3, 4]:
    label = QUARTILE_LABELS[q]
    mask  = tte_clean['Z_Quartile'] == q
    sub   = tte_clean[mask]
    kmf   = KaplanMeierFitter()
    kmf.fit(sub['Duration'], event_observed=sub['Event'], label=label)
    kmf.plot_survival_function(
        ax=ax_q, ci_show=True,
        color=PALETTE[label], linewidth=2.0
    )
    kmf_list[label] = kmf
    print(f"  {label:20s}: median={kmf.median_survival_time_} yrs  "
          f"(n={len(sub)}, events={sub['Event'].sum()})")

# -------------------------------------------------------------------------
# NEW: PRINT YEARLY SURVIVAL PROBABILITIES BY QUARTILE (Q1-Q4)
# -------------------------------------------------------------------------
print("\n  Probability of Remaining Healthy (Survival) by Year across Z*-Score Quartiles:")
print("  " + "-" * 65)
print("  Year |     Q1 (High) |            Q2 |            Q3 |      Q4 (Low)")
print("  " + "-" * 65)

max_duration = int(tte_clean['Duration'].max())
for year in range(1, max_duration + 1):
    p = {}
    for q in [1, 2, 3, 4]:
        label = QUARTILE_LABELS[q]
        # Get the specific probability for this quartile at this year
        p[q] = kmf_list[label].predict(year) * 100
        
    print(f"    {year:2d} |        {p[1]/100:6.2f} |        {p[2]/100:6.2f} |        {p[3]/100:6.2f} |        {p[4]/100:6.2f}")
print("  " + "-" * 65 + "\n")
# -------------------------------------------------------------------------

ax_q.set_title('Kaplan-Meier: Survival by Revised Altman Z*-Score Quartile\n'
               'Q1 = lowest Z* (highest risk),  Q4 = highest Z* (lowest risk)',
               fontsize=11)
ax_q.set_xlabel('Years since first observation', fontsize=11)
ax_q.set_ylabel('P(remaining non-distressed)', fontsize=11)
ax_q.set_ylim(0, 1.05)
ax_q.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
ax_q.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('survival_km_zscore_groups.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved -> 'survival_km_zscore_groups.png'")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: LOG-RANK TEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: LOG-RANK TEST")
print("=" * 65)

mv = multivariate_logrank_test(
    tte_clean['Duration'],
    tte_clean['Z_Group'],
    event_observed=tte_clean['Event']
)
print(f"  Test statistic : {mv.test_statistic:.4f}")
print(f"  p-value        : {mv.p_value:.6f}")

if mv.p_value < 0.001:
    sig_label = "p < 0.001 ***"
elif mv.p_value < 0.01:
    sig_label = f"p = {mv.p_value:.4f} **"
elif mv.p_value < 0.05:
    sig_label = f"p = {mv.p_value:.4f} *"
else:
    sig_label = f"p = {mv.p_value:.4f} (not significant at 0.05)"

print(f"  Result         : {sig_label}")

# Pairwise: Q1 vs Q4 is the headline comparison for the paper
print("\n  Pairwise log-rank tests:")
for (qa, qb) in [(1, 4), (1, 2), (3, 4)]:
    la, lb = QUARTILE_LABELS[qa], QUARTILE_LABELS[qb]
    ga = tte_clean[tte_clean['Z_Quartile'] == qa]
    gb = tte_clean[tte_clean['Z_Quartile'] == qb]
    res = logrank_test(
        ga['Duration'], gb['Duration'],
        event_observed_A=ga['Event'],
        event_observed_B=gb['Event']
    )
    sig = ('***' if res.p_value < 0.001 else
           ('**' if res.p_value < 0.01 else
            ('*' if res.p_value < 0.05 else 'n.s.')))
    print(f"  {la} vs {lb}: p={res.p_value:.4f} {sig}")

# Log-rank panel figure
fig_lr, axes_lr = plt.subplots(1, 2, figsize=(13, 5))

ax_l = axes_lr[0]
for q in [1, 2, 3, 4]:
    label = QUARTILE_LABELS[q]
    kmf_list[label].plot_survival_function(
        ax=ax_l, ci_show=False,
        color=PALETTE[label], linewidth=2.0
    )
ax_l.set_title('KM Curves by Z*-Score Quartile', fontsize=11)
ax_l.set_xlabel('Years')
ax_l.set_ylabel('P(non-distressed)')
ax_l.set_ylim(0, 1.05)
ax_l.grid(True, alpha=0.3)

ax_r = axes_lr[1]
q_data = []
for q in [1, 2, 3, 4]:
    mask = tte_clean['Z_Quartile'] == q
    sub  = tte_clean[mask]
    q_data.append({
        'label': QUARTILE_LABELS[q].replace(' (highest risk)', '').replace(' (lowest risk)', ''),
        'rate' : sub['Event'].mean() * 100,
        'color': list(PALETTE.values())[q - 1],
    })

bars = ax_r.bar(
    [d['label'] for d in q_data],
    [d['rate']  for d in q_data],
    color=[d['color'] for d in q_data],
    width=0.5, alpha=0.8,
    edgecolor='#444441', linewidth=0.5
)
for bar, d in zip(bars, q_data):
    ax_r.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        f"{d['rate']:.1f}%",
        ha='center', va='bottom', fontsize=9
    )
ax_r.set_title(f'Distress Event Rate by Quartile\nLog-rank: {sig_label}', fontsize=11)
ax_r.set_ylabel('% firms that became distressed', fontsize=10)
ax_r.set_xlabel('Z*-Score Quartile', fontsize=10)
ax_r.set_ylim(0, max(d['rate'] for d in q_data) * 1.3)
ax_r.grid(True, axis='y', alpha=0.3)
ax_r.spines['top'].set_visible(False)
ax_r.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('survival_km_logrank.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved -> 'survival_km_logrank.png'")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: COX PROPORTIONAL HAZARDS MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: COX PROPORTIONAL HAZARDS MODEL")
print("=" * 65)
print(f"  Features: {COX_FEATURES}")
print(f"  Selection rationale: one per financial health dimension,")
print(f"  pairwise correlations < 0.70, all established in distress literature.")

cox_df = tte[['Duration', 'Event'] + COX_FEATURES].copy()

# Winsorize at 1st-99th percentile per feature
# Prevents extreme outliers from distorting hazard ratio estimates
for col in COX_FEATURES:
    q01 = cox_df[col].quantile(0.01)
    q99 = cox_df[col].quantile(0.99)
    cox_df[col] = cox_df[col].clip(q01, q99)

n_before = len(cox_df)
cox_df.dropna(inplace=True)
print(f"\n  Cox sample  : {len(cox_df):,} firms "
      f"(dropped {n_before - len(cox_df):,} with missing values)")
print(f"  Cox events  : {cox_df['Event'].sum():,} "
      f"({cox_df['Event'].mean()*100:.1f}%)")
print(f"  EPV = {cox_df['Event'].sum()} / {len(COX_FEATURES)} = "
      f"{cox_df['Event'].sum()/len(COX_FEATURES):.0f}  (minimum recommended: 10)")

cph = CoxPHFitter(penalizer=0.1)  # L2 penalty for numerical stability
cph.fit(cox_df, duration_col='Duration', event_col='Event', show_progress=False)

print("\n  Results:")
print("  HR > 1 = accelerates distress  |  HR < 1 = protective (delays distress)")
print()
cph.print_summary(decimals=4, style="ascii")

c_idx = cph.concordance_index_
print(f"\n  Concordance index (C-statistic): {c_idx:.4f}")
print(f"  Interpretation: model correctly ranks {c_idx*100:.1f}% of firm pairs")
print(f"  by their actual time to distress.")
print(f"  Benchmark: random=0.50 | acceptable=0.65 | good=0.70+")

# Save results
cox_out = cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z',
                        'p', 'exp(coef) lower 95%',
                        'exp(coef) upper 95%']].copy()
cox_out.columns = ['Beta', 'Hazard_Ratio', 'SE', 'z_stat',
                   'p_value', 'HR_lower_95', 'HR_upper_95']
cox_out.index.name = 'Feature'
cox_out['Significant'] = cox_out['p_value'].apply(
    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else
    ('*' if p < 0.05 else ''))
)
cox_out['Category'] = cox_out.index.map({
    'Current Ratio'                : 'Liquidity',
    'Net Profit Margin'            : 'Profitability',
    'Interest Coverage'            : 'Debt service',
    'Total Debt To Capitalization' : 'Leverage',
    'Income Quality'               : 'Accruals quality',
})
cox_out = cox_out.sort_values('Hazard_Ratio', ascending=False)
cox_out.to_csv('survival_cox_results.csv')
print(f"\n  Saved -> 'survival_cox_results.csv'")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: COX FOREST PLOT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: COX FOREST PLOT")
print("=" * 65)

plot_df = cox_out.reset_index().sort_values('Hazard_Ratio')

fig_c, ax_c = plt.subplots(figsize=(9, max(4, len(plot_df) * 0.7 + 1.5)))
y_pos   = np.arange(len(plot_df))
colours = ['#D85A30' if hr > 1 else '#1D9E75' for hr in plot_df['Hazard_Ratio']]

ax_c.barh(
    y_pos, plot_df['Hazard_Ratio'] - 1, left=1,
    xerr=[plot_df['Hazard_Ratio'] - plot_df['HR_lower_95'],
          plot_df['HR_upper_95']  - plot_df['Hazard_Ratio']],
    color=colours, alpha=0.8, height=0.55,
    error_kw={'ecolor': '#444441', 'capsize': 4, 'elinewidth': 1.0}
)
ax_c.axvline(1, color='#444441', linewidth=1.0, linestyle='--', alpha=0.8)

# Category labels on the left side
for i, (_, row) in enumerate(plot_df.iterrows()):
    cat = row.get('Category', '')
    ax_c.text(-0.01, i, f"[{cat}]", ha='right', va='center',
              fontsize=7.5, color='#888780',
              transform=ax_c.get_yaxis_transform())

# Significance markers on the right
for i, (_, row) in enumerate(plot_df.iterrows()):
    if row['Significant']:
        ax_c.text(0.99, i, row['Significant'],
                  ha='right', va='center', fontsize=10,
                  transform=ax_c.get_yaxis_transform())

ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(plot_df['Feature'].tolist(), fontsize=9)
ax_c.set_xlabel('Hazard Ratio  (HR = 1 : no effect on time-to-distress)',
                fontsize=10)
ax_c.set_title(
    f'Cox PH Model: Hazard Ratios with 95% CI\n'
    f'C-index = {c_idx:.3f}  |  '
    f'* p<0.05   ** p<0.01   *** p<0.001',
    fontsize=11
)
ax_c.grid(True, axis='x', alpha=0.3)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

red_p  = mpatches.Patch(color='#D85A30', alpha=0.8,
                         label='HR > 1  (accelerates distress)')
teal_p = mpatches.Patch(color='#1D9E75', alpha=0.8,
                         label='HR < 1  (protective / delays distress)')
ax_c.legend(handles=[red_p, teal_p], fontsize=9,
            loc='lower right', framealpha=0.8)

plt.tight_layout()
plt.savefig('survival_cox_forest_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved -> 'survival_cox_forest_plot.png'")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("COMPLETE")
print("=" * 65)
print(f"""
  Kaplan-Meier
    Firms                 : {len(tte):,}
    Distress events       : {tte['Event'].sum():,} ({tte['Event'].mean()*100:.1f}%)
    Censored              : {(tte['Event']==0).sum():,}
    Median survival       : {kmf_overall.median_survival_time_} years
    Log-rank (4 groups)   : {sig_label}

  Cox PH Model
    Sample                : {len(cox_df):,} firms / {cox_df['Event'].sum()} events
    Features              : {len(COX_FEATURES)} (one per financial dimension)
    EPV                   : {cox_df['Event'].sum()/len(COX_FEATURES):.0f} (well above minimum of 10)
    Concordance index     : {c_idx:.4f}

  Output files
    survival_time_to_distress.csv
    survival_km_overall.png
    survival_km_zscore_groups.png
    survival_km_logrank.png
    survival_cox_results.csv
    survival_cox_forest_plot.png
""")