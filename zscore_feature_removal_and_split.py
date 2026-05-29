"""
Altman Z*-Score — Feature Removal and Train-Test Split
=======================================================
INPUT  : nse_altman_corrected.csv   (output of altman_z_score_corrected.py)
OUTPUT : nse_altman_model_ready.csv  (full dataset, final features + label)
         X_train_z.csv               (75% stratified train, scaled)
         X_test_z.csv                (25% stratified test,  scaled)
         y_train_z.csv               (train labels)
         y_test_z.csv                (test labels)

WHY WE REMOVE Z-SCORE INPUT COLUMNS:
  The Revised Altman Z*-Score label was derived from 4 components,
  each built from specific dataset columns:

    X1 (WC/TA)       ← Working Capital, Tangible Asset Value
    X2 (RE/TA proxy) ← Return On Assets
    X3 (EBIT/TA)     ← EBIT Per Revenue, Asset Turnover
    X4 (MktCap/TL)   ← PB Ratio, Debt Ratio

  If these columns remain in the feature set, ML models can trivially
  reconstruct Z* and predict the label without learning genuine
  financial distress signals.  This is data leakage.

  We also remove algebraic derivatives — columns that are
  mathematically equivalent or near-identical transformations of
  the direct inputs (e.g. Debt To Assets = Debt Ratio; ROE ≈ ROA).

  Total removed: 25 columns across all 4 components.
  Remaining for modelling: 73 columns → after multicollinearity
  removal in the preprocessing step, expect ~40-55 final features.

WHY SCALING IS DONE HERE (NOT IN THE PREPROCESSING SCRIPT):
  The preprocessing script (altman_z_score_corrected.py) intentionally
  omits Min-Max scaling. Scaling must happen AFTER the train-test split
  to prevent data leakage — the scaler must be fitted on training data
  only, then applied to both sets using those training-derived bounds.

WHY STRATIFIED SPLIT:
  Preserves the distressed/non-distressed ratio identically in both
  train and test sets, ensuring the test set is a fair evaluation sample.

WHY TEST SET IS NEVER SCALED WITH TEST DATA:
  scaler.fit(X_train) → scaler.transform(X_train) and scaler.transform(X_test)
  The test set is transformed using training bounds, never its own bounds.
"""

import pandas as pd
import numpy as np
try:
    from sklearn.model_selection import train_test_split
except Exception:
    # Lightweight fallback for environments without scikit-learn installed.
    def train_test_split(X, y, test_size=0.25, random_state=None, stratify=None):
        """Minimal replacement of sklearn.model_selection.train_test_split.
        Returns X_train, X_test, y_train, y_test. Supports numpy arrays and pandas objects.
        Supports fractional test_size (0-1) or absolute number >=1.
        Supports simple stratify by label array.
        """
        if hasattr(X, 'shape'):
            n = X.shape[0]
        else:
            n = len(X)

        rng = np.random.RandomState(random_state)

        # compute test count
        if 0 < test_size < 1:
            n_test = int(np.floor(n * test_size))
        else:
            n_test = int(test_size)

        indices = np.arange(n)

        if stratify is None:
            rng.shuffle(indices)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]
        else:
            # stratified split by label values
            labels = np.array(stratify)
            train_idx = []
            test_idx = []
            for lbl in np.unique(labels):
                lbl_idx = indices[labels == lbl]
                rng.shuffle(lbl_idx)
                k = int(np.floor(len(lbl_idx) * (n_test / n)))
                # ensure at least one goes to test if possible
                k = max(k, 1) if len(lbl_idx) > 1 and k == 0 else k
                test_idx.extend(lbl_idx[:k].tolist())
                train_idx.extend(lbl_idx[k:].tolist())
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)

        def subset(a, idx):
            if hasattr(a, 'iloc'):
                return a.iloc[idx]
            else:
                return a[idx]

        X_train = subset(X, train_idx)
        X_test = subset(X, test_idx)
        y_train = subset(y, train_idx)
        y_test = subset(y, test_idx)

        return X_train, X_test, y_train, y_test
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: LOADING CLEANED ALTMAN DATASET")
print("=" * 65)

df = pd.read_csv('nse_altman_corrected.csv', low_memory=False)
print(f"  Loaded : {df.shape[0]:,} rows × {df.shape[1]} columns")

vc = df['Financial_Distress'].value_counts()
print(f"  Distressed   (1): {vc[1]:,}  ({vc[1]/len(df)*100:.1f}%)")
print(f"  Non-distressed(0): {vc[0]:,}  ({vc[0]/len(df)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: REMOVING Z-SCORE INPUT COLUMNS (preventing data leakage)")
print("=" * 65)

# Every column listed here is a direct input or algebraic derivative
# of one of the 4 Revised Altman Z*-Score components.
# Keeping them would let models reconstruct Z* and trivially predict
# the label rather than learning genuine distress patterns.

ZSCORE_INPUT_COLS = {
    # ── X1 = Working Capital / Tangible Asset Value ───────────────────────────
    'Working Capital'             : 'X1 direct input (numerator)',
    'Tangible Asset Value'        : 'X1 direct input (denominator)',
    'Net Current Asset Value'     : 'X1 derivative  (≈ WC minus long-term debt)',
    'Tangible Book Value Per Share': 'X1 derivative  (per-share form of Tangible Asset Value)',
    'Book Value Per Share'        : 'X1 derivative  (correlated with Tangible Asset Value)',
    'Shareholders Equity Per Share': 'X1 derivative  (= Book Value Per Share)',

    # ── X2 = Return On Assets (RE/TA proxy) ───────────────────────────────────
    'Return On Assets'            : 'X2 direct input (= ROA)',
    'Return On Tangible Assets'   : 'X2 derivative  (same formula, tangible denominator)',
    'Return On Capital Employed'  : 'X2 derivative  (EBIT/Capital — uses same EBIT & asset base)',
    'ROIC'                        : 'X2 derivative  (Net Income / Invested Capital ≈ ROA)',
    'ROE'                         : 'X2 derivative  (Net Income / Equity, correlated with ROA)',
    'Return On Equity'            : 'X2 derivative  (= ROE, duplicate column)',

    # ── X3 = EBIT Per Revenue × Asset Turnover (= EBIT/TA exactly) ───────────
    'EBIT Per Revenue'            : 'X3 direct input (EBIT/Sales)',
    'Asset Turnover'              : 'X3 direct input (Sales/TA)',
    'Operating Profit Margin'     : 'X3 derivative  (≈ EBIT Per Revenue, near-identical)',
    'Fixed Asset Turnover'        : 'X3 derivative  (Sales/Fixed Assets, correlated with AT)',
    # Note: Return On Capital Employed also uses EBIT — already removed above

    # ── X4 = PB Ratio × (1 − Debt Ratio) / Debt Ratio (= MktCap/TL exactly) ──
    'PB Ratio'                    : 'X4 direct input (MktCap/Book Value)',
    'Debt Ratio'                  : 'X4 direct input (TL/TA)',
    'PTB Ratio'                   : 'X4 derivative  (= PB Ratio, duplicate)',
    'Price Book Value Ratio'      : 'X4 derivative  (= PB Ratio, duplicate)',
    'Price To Book Ratio'         : 'X4 derivative  (= PB Ratio, duplicate)',
    'Debt To Assets'              : 'X4 derivative  (= Debt Ratio, duplicate)',
    'Debt Equity Ratio'           : 'X4 derivative  (= D/E, directly derived from Debt Ratio)',
    'Debt To Equity'              : 'X4 derivative  (= Debt Equity Ratio, duplicate)',
    'Company Equity Multiplier'   : 'X4 derivative  (= TA/Equity = 1/(1−DR), derived from DR)',
}

print(f"\n  Columns to remove: {len(ZSCORE_INPUT_COLS)}\n")
for col, reason in ZSCORE_INPUT_COLS.items():
    in_df  = col in df.columns
    status = "REMOVED" if in_df else "already absent"
    print(f"  [{status:14s}]  {col}")
    print(f"                       Reason: {reason}")

# Build the feature set: everything except meta, score, and leakage columns
NON_FEATURE = ['Year', 'Stock', 'Z_Score', 'Altman_Zone', 'Financial_Distress']
REMOVE      = [c for c in ZSCORE_INPUT_COLS if c in df.columns]

feature_cols = [
    c for c in df.columns
    if c not in NON_FEATURE and c not in REMOVE
]

print(f"\n  Features before removal : {len(df.columns) - len(NON_FEATURE)}")
print(f"  Z-Score inputs removed  : {len(REMOVE)}")
print(f"  Features after removal  : {len(feature_cols)}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: FINAL FEATURE LIST")
print("=" * 65)

# Map surviving features to financial categories for transparency
CATEGORIES = {
    'Liquidity': [
        'Current Ratio', 'Quick Ratio', 'Cash Ratio', 'Cash Per Share',
        'Cash Flow Coverage Ratios', 'Cash Flow To Debt Ratio',
        'Cash Conversion Cycle', 'Short Term Coverage Ratios',
        'Days Of Sales Outstanding', 'Days Sales Outstanding',
    ],
    'Efficiency': [
        'Average Inventory', 'Average Payables', 'Average Receivables',
        'Inventory Turnover', 'Receivables Turnover', 'Payables Turnover',
        'Days Of Inventory On Hand', 'Days Of Inventory Outstanding',
        'Days Of Payables Outstanding', 'Days Payables Outstanding',
        'Operating Cycle',
    ],
    'Solvency / Coverage': [
        'Long Term Debt To Capitalization', 'Total Debt To Capitalization',
        'Interest Coverage', 'Interest Debt Per Share', 'Invested Capital',
        'Net Debt To Ebitda', 'Capital Expenditure Coverage Ratio',
        'Dividend Paid And Capex Coverage Ratio',
    ],
    'Profitability': [
        'Gross Profit Margin', 'Net Profit Margin', 'Pretax Profit Margin',
        'EBT Per EBIT', 'Net Income Per EBT', 'Net Income Per Share',
        'Effective Tax Rate', 'Earnings Yield', 'Income Quality',
    ],
    'Cash Flow': [
        'Free Cash Flow Per Share', 'Operating Cash Flow Per Share',
        'Free Cash Flow Operating Cash Flow Ratio', 'Free Cash Flow Yield',
        'Capex Per Share', 'Capex To Depreciation',
        'Capex To Operating Cash Flow', 'Capex To Revenue',
        'Dividend Payout Ratio', 'Payout Ratio',
        'Stock Based Compensation To Revenue',
        'Research And Ddevelopement To Revenue',
    ],
    'Valuation / Market': [
        'PE Ratio', 'Price Earnings Ratio', 'Price Earnings To Growth Ratio',
        'EV To Sales', 'Enterprise Value', 'Enterprise Value Over Ebitda',
        'Ev To Free Cash Flow', 'Ev To Operating Cash Flow',
        'PFCF Ratio', 'POCF Ratio', 'Price Cash Flow Ratio',
        'Price Sales Ratio', 'Price To Sales Ratio',
        'Price To Free Cash Flows Ratio', 'Price To Operating Cash Flows Ratio',
        'Price Fair Value', 'Market Cap', 'Dividend Yield',
        'Graham Net Net', 'Graham Number',
        'Revenue Per Share',
    ],
    'Other': [
        'Intangibles To Total Assets',
        'Sales General And Administrative To Revenue',
    ],
}

print(f"\n  {len(feature_cols)} independent features remaining:\n")
categorised = []
for cat, cols in CATEGORIES.items():
    present = [c for c in cols if c in feature_cols]
    if present:
        print(f"  {cat}  ({len(present)} features):")
        for c in present:
            print(f"    - {c}")
            categorised.append(c)

uncategorised = [c for c in feature_cols if c not in categorised]
if uncategorised:
    print(f"\n  Uncategorised ({len(uncategorised)}):")
    for c in uncategorised:
        print(f"    - {c}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: TRAIN-TEST SPLIT (75% / 25%, stratified)")
print("=" * 65)

X = df[feature_cols].copy()
y = df['Financial_Distress'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.25,
    random_state = 42,
    stratify     = y          # preserves class ratio in both sets
)

print(f"""
  Split parameters:
    Test size    : 25%
    Random state : 42  (reproducible)
    Stratified   : Yes

  ┌───────────────────────────────────────────────────────────┐
  │                TRAIN SET          TEST SET                │
  ├───────────────────────────────────────────────────────────┤
  │  Rows        : {len(X_train):>8,}         {len(X_test):>8,}           │
  │  Distressed  : {(y_train==1).sum():>8,} ({(y_train==1).mean()*100:.1f}%)  {(y_test==1).sum():>8,} ({(y_test==1).mean()*100:.1f}%)     │
  │  Non-dist.   : {(y_train==0).sum():>8,} ({(y_train==0).mean()*100:.1f}%)  {(y_test==0).sum():>8,} ({(y_test==0).mean()*100:.1f}%)     │
  │  Features    : {len(feature_cols):>8,}         {len(feature_cols):>8,}           │
  └───────────────────────────────────────────────────────────┘

  Stratification check:
    Original distress %  : {y.mean()*100:.2f}%
    Train distress %     : {y_train.mean()*100:.2f}%
    Test  distress %     : {y_test.mean()*100:.2f}%
""")


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 5: MIN-MAX SCALING (fit on train, apply to both)")
print("=" * 65)

# Fit ONLY on training data — test data must never influence the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_cols,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_cols,
    index=X_test.index
)

print(f"  Scaler fitted on X_train ({len(X_train):,} rows) only.")
print(f"  X_train scaled: shape {X_train_scaled.shape}")
print(f"  X_test  scaled: shape {X_test_scaled.shape}")
print(f"  Sample range check (train):")
for c in feature_cols[:3]:
    print(f"    {c}: min={X_train_scaled[c].min():.4f}, max={X_train_scaled[c].max():.4f}")
print(f"  Test set is transformed using TRAINING bounds — not refitted.")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: SAVING ALL OUTPUTS")
print("=" * 65)

# Full model-ready dataset (for reference / inspection)
df_model = df[['Year', 'Stock'] + feature_cols + ['Z_Score', 'Altman_Zone', 'Financial_Distress']].copy()
df_model.to_csv('nse_altman_model_ready.csv', index=False)
print(f"  Saved 'nse_altman_model_ready.csv'  → {df_model.shape[0]:,} rows × {df_model.shape[1]} cols")

# Train / test splits (scaled)
X_train_scaled.to_csv('X_train_z.csv', index=False)
X_test_scaled.to_csv('X_test_z.csv',  index=False)
y_train.to_csv('y_train_z.csv', index=False, header=True)
y_test.to_csv('y_test_z.csv',  index=False, header=True)

print(f"  Saved 'X_train_z.csv'               → {X_train_scaled.shape[0]:,} rows × {X_train_scaled.shape[1]} cols")
print(f"  Saved 'X_test_z.csv'                → {X_test_scaled.shape[0]:,} rows × {X_test_scaled.shape[1]} cols")
print(f"  Saved 'y_train_z.csv'               → {len(y_train):,} labels")
print(f"  Saved 'y_test_z.csv'                → {len(y_test):,} labels")

print(f"""
  ┌───────────────────────────────────────────────────────────┐
  │  SUMMARY — READY FOR MODELLING                            │
  ├───────────────────────────────────────────────────────────┤
  │  Final features                 : {len(feature_cols):>3}                    │
  │  Z-Score inputs removed         : {len(REMOVE):>3}  (leakage prevention) │
  │  Train rows                     : {len(X_train):>6,}                 │
  │  Test rows                      : {len(X_test):>6,}                 │
  │  Scaling                        : fit on train, apply to both  │
  │                                                           │
  │  NEXT STEP:                                               │
  │  Run zscore_smote_to_ann_pipeline.py                      │
  └───────────────────────────────────────────────────────────┘
""")
print("Done.\n")
