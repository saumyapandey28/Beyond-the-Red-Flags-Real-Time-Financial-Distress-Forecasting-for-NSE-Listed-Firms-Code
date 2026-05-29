import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.model_selection    import GridSearchCV, StratifiedKFold
from sklearn.feature_selection  import RFECV
from sklearn.linear_model       import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier
from xgboost                    import XGBClassifier
from sklearn.svm                import SVC
from sklearn.neural_network     import MLPClassifier
from sklearn.neighbors          import NearestNeighbors
from sklearn.utils              import resample
from sklearn.metrics            import (accuracy_score, precision_score,
                                        recall_score, f1_score,
                                        roc_auc_score, classification_report,
                                        confusion_matrix)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred, y_prob=None, train_acc=None):
    """Compute and return all evaluation metrics as a dict."""
    row = {
        "Model"    : name,
        "Train Acc": round(train_acc, 4) if train_acc is not None else "-",
        "Test Acc" : round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "F1"       : round(f1_score(y_true, y_pred,        zero_division=0), 4),
        "AUC"      : round(roc_auc_score(y_true, y_prob), 4) if y_prob is not None else "-",
    }
    return row

def print_table(rows, title):
    """Pretty-print results as a table and return as DataFrame."""
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    df_t = pd.DataFrame(rows)
    print(df_t.to_string(index=False))
    return df_t


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 75)
print("LOADING DATA")
print("=" * 75)

X_train = pd.read_csv("C:/Users/hardi/Downloads/Documents/X_train_z.csv")
X_test  = pd.read_csv("C:/Users/hardi/Downloads/Documents/X_test_z.csv")
y_train = pd.read_csv("C:/Users/hardi/Downloads/Documents/y_train_z.csv").squeeze()
y_test  = pd.read_csv("C:/Users/hardi/Downloads/Documents/y_test_z.csv").squeeze()

feature_names  = X_train.columns.tolist()
X_train_arr    = X_train.values
X_test_arr     = X_test.values
y_train_arr    = y_train.values
y_test_arr     = y_test.values

print(f"  X_train  : {X_train_arr.shape}  |  y_train : {len(y_train_arr)}")
print(f"  X_test   : {X_test_arr.shape}   |  y_test  : {len(y_test_arr)}")
print(f"  Features : {len(feature_names)}")
print(f"  Train class dist → 0: {(y_train_arr==0).sum()}  "
      f"1: {(y_train_arr==1).sum()}  "
      f"({y_train_arr.mean()*100:.1f}% distressed)")
print(f"  Test  class dist → 0: {(y_test_arr==0).sum()}  "
      f"1: {(y_test_arr==1).sum()}  "
      f"({y_test_arr.mean()*100:.1f}% distressed)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SMOTE + ENN  (training set ONLY)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 1: SMOTE + ENN  (training set only)")
print("=" * 75)

def smote(X, y, minority_class=1, k=5, random_state=42):
    """Oversample minority class to match majority class size."""
    rng   = np.random.RandomState(random_state)
    X_min = X[y == minority_class]
    X_maj = X[y != minority_class]
    n_to_gen = len(X_maj) - len(X_min)
    if n_to_gen <= 0:
        return X, y

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, indices = nn.kneighbors(X_min)   # includes self

    synthetic = []
    for i in range(n_to_gen):
        idx    = i % len(X_min)
        nn_idx = indices[idx, 1:]       # exclude self
        chosen = rng.choice(nn_idx)
        lam    = rng.uniform(0, 1)
        synthetic.append(X_min[idx] + lam * (X_min[chosen] - X_min[idx]))

    X_syn = np.array(synthetic)
    y_syn = np.full(len(X_syn), minority_class)
    return np.vstack([X, X_syn]), np.hstack([y, y_syn])


def enn_clean(X, y, k=3):
    """Remove samples misclassified by their k nearest neighbours."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, indices = nn.kneighbors(X)
    keep = [i for i in range(len(X))
            if np.bincount(y[indices[i, 1:]]).argmax() == y[i]]
    return X[keep], y[keep]


X_smote, y_smote = smote(X_train_arr, y_train_arr, k=5, random_state=42)
X_res,   y_res   = enn_clean(X_smote, y_smote, k=3)

print(f"  Before SMOTE+ENN : {len(y_train_arr):,} samples  "
      f"(0={(y_train_arr==0).sum()}  1={(y_train_arr==1).sum()})")
print(f"  After SMOTE      : {len(y_smote):,} samples  "
      f"(0={(y_smote==0).sum()}  1={(y_smote==1).sum()})")
print(f"  After ENN clean  : {len(y_res):,} samples  "
      f"(0={(y_res==0).sum()}  1={(y_res==1).sum()})")
print(f"  Test set         : UNCHANGED — {len(y_test_arr)} rows, "
      f"(0={(y_test_arr==0).sum()}  1={(y_test_arr==1).sum()})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — RFECV FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 2: RFECV Feature Selection  (resampled training set)")
print("=" * 75)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfecv_estimators = {
    "LR" : LogisticRegression(solver="lbfgs", penalty="l2", C=1.0,
                               max_iter=1000, class_weight="balanced",
                               random_state=42),
    "DT" : DecisionTreeClassifier(max_depth=3, criterion="gini",
                                  random_state=42),
    "RF" : RandomForestClassifier(n_estimators=100, max_depth=3,
                                   criterion="gini", random_state=42),
    "SVM": SVC(kernel="linear", C=1.0, probability=True, random_state=42),
}

rfecv_masks   = {}
rfecv_summary = []

for name, est in rfecv_estimators.items():
    print(f"  Running RFECV for {name} ...", end=" ", flush=True)
    rfecv = RFECV(estimator=est, step=1, cv=cv5,
                  scoring="f1", min_features_to_select=1, n_jobs=-1)
    rfecv.fit(X_res, y_res)
    rfecv_masks[name] = rfecv.support_
    best_cv_f1 = rfecv.cv_results_["mean_test_score"].max()
    rfecv_summary.append({
        "Estimator"       : name,
        "Optimal Features": rfecv.n_features_,
        "Best CV F1"      : round(best_cv_f1, 4),
    })
    print(f"optimal features: {rfecv.n_features_:3d}  CV F1: {best_cv_f1:.4f}")

# Union mask: keep a feature if selected by ANY estimator
union_mask = np.zeros(len(feature_names), dtype=bool)
for mask in rfecv_masks.values():
    union_mask |= mask

rfecv_feature_names = [f for f, m in zip(feature_names, union_mask) if m]
X_train_rfecv       = X_res[:, union_mask]
X_test_rfecv        = X_test_arr[:, union_mask]

print(f"\n  Union of selected features : {len(rfecv_feature_names)}")
print(f"  Selected: {rfecv_feature_names}")
print_table(rfecv_summary, "RFECV Summary — Altman Z*-Score")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EXPERIMENT 1: PREDICTIVE MODELING (full feature set)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("EXPERIMENT 1: Predictive Modeling — Full Feature Set")
print("=" * 75)

param_grids = {
    "LR" : {"C": [0.1, 1, 10, 100],
             "max_iter": [100, 500],
             "solver": ["lbfgs", "newton-cg"],
             "penalty": ["l2"]},
    "DT" : {"max_depth": [3, 6, 9, 12],
             "min_samples_leaf": [1, 2, 4],
             "splitter": ["best"]},
    "RF" : {"n_estimators": [10, 50, 100],
             "max_depth": [3, 4, 6],
             "min_samples_leaf": [1, 2]},
    "SVM": {"C": [1, 10, 100],
             "kernel": ["rbf"],
             "gamma": [0.1, 1],
             "degree": [2]},
    "XGB": {"n_estimators": [50, 100],
             "max_depth": [3, 4],
             "learning_rate": [0.05, 0.1],
             "subsample": [0.8, 1.0]},
}

base_estimators = {
    "LR" : LogisticRegression(random_state=42),
    "DT" : DecisionTreeClassifier(random_state=42),
    "RF" : RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGB": XGBClassifier(random_state=42, eval_metric="logloss"),
}

exp1_results = []
exp1_models  = {}

for name, est in base_estimators.items():
    print(f"  Training {name} ...", end=" ", flush=True)
    gs = GridSearchCV(est, param_grids[name], cv=5,
                      scoring="f1", n_jobs=-1, refit=True)
    gs.fit(X_res, y_res)
    best = gs.best_estimator_

    tr_acc = accuracy_score(y_res, best.predict(X_res))
    y_pred = best.predict(X_test_arr)
    y_prob = (best.predict_proba(X_test_arr)[:, 1]
              if hasattr(best, "predict_proba") else None)

    row = evaluate(name, y_test_arr, y_pred, y_prob, tr_acc)
    exp1_results.append(row)
    exp1_models[name] = best
    print(f"Test Acc: {row['Test Acc']}  F1: {row['F1']}  AUC: {row['AUC']}")

df_exp1 = print_table(exp1_results, "Experiment 1 Results — Predictive Modeling (Altman Z*)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — EXPERIMENT 2: RFECV MODELS (RFECV-selected features only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("EXPERIMENT 2: RFECV Models — RFECV-Selected Features Only")
print("=" * 75)

exp2_results = []
exp2_models  = {}

for name, est in base_estimators.items():
    print(f"  Training {name} (RFECV features) ...", end=" ", flush=True)
    gs = GridSearchCV(est, param_grids[name], cv=5,
                      scoring="f1", n_jobs=-1, refit=True)
    gs.fit(X_train_rfecv, y_res)
    best = gs.best_estimator_

    tr_acc = accuracy_score(y_res, best.predict(X_train_rfecv))
    y_pred = best.predict(X_test_rfecv)
    y_prob = (best.predict_proba(X_test_rfecv)[:, 1]
              if hasattr(best, "predict_proba") else None)

    row = evaluate(name, y_test_arr, y_pred, y_prob, tr_acc)
    exp2_results.append(row)
    exp2_models[name] = best
    print(f"Test Acc: {row['Test Acc']}  F1: {row['F1']}  AUC: {row['AUC']}")

df_exp2 = print_table(exp2_results, "Experiment 2 Results — RFECV Models (Altman Z*)")

# Side-by-side comparison
print("\n  Experiment 1 vs Experiment 2 Comparison:")
comp = pd.DataFrame({
    "Model"   : [r["Model"] for r in exp1_results],
    "Exp1 Acc": [r["Test Acc"] for r in exp1_results],
    "Exp2 Acc": [r["Test Acc"] for r in exp2_results],
    "Exp1 F1" : [r["F1"] for r in exp1_results],
    "Exp2 F1" : [r["F1"] for r in exp2_results],
    "Exp1 AUC": [r["AUC"] for r in exp1_results],
    "Exp2 AUC": [r["AUC"] for r in exp2_results],
})
print(comp.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EXPERIMENT 3: CART BOOTSTRAPPING (1:1 to 10:1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("EXPERIMENT 3: CART Bootstrapping — Ratios 1:1 to 10:1")
print("=" * 75)
print("  (Non-distressed : Distressed ratio)  |  Uses original train, not SMOTE")

dist_idx     = np.where(y_train_arr == 1)[0]
non_dist_idx = np.where(y_train_arr == 0)[0]
n_dist       = len(dist_idx)

cart_results = []

for ratio in range(1, 11):
    n_non = min(n_dist * ratio, len(non_dist_idx))

    sampled_non = resample(non_dist_idx, n_samples=n_non,
                           replace=True, random_state=42)
    idx_boot    = np.concatenate([dist_idx, sampled_non])
    X_boot      = X_train_arr[idx_boot]
    y_boot      = y_train_arr[idx_boot]

    cart = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2,
        criterion="gini", random_state=42
    )
    cart.fit(X_boot, y_boot)

    # Train metrics (on bootstrapped sample)
    y_boot_pred = cart.predict(X_boot)
    y_boot_prob = cart.predict_proba(X_boot)[:, 1]
    tr_acc  = accuracy_score(y_boot, y_boot_pred)
    tr_f1   = f1_score(y_boot, y_boot_pred, zero_division=0)
    tr_auc  = roc_auc_score(y_boot, y_boot_prob)

    # Test metrics (original unmodified test set)
    y_te_pred = cart.predict(X_test_arr)
    y_te_prob = cart.predict_proba(X_test_arr)[:, 1]
    te_acc  = accuracy_score(y_test_arr, y_te_pred)
    te_f1   = f1_score(y_test_arr, y_te_pred, zero_division=0)
    te_auc  = roc_auc_score(y_test_arr, y_te_prob)

    cart_results.append({
        "Sample"   : ratio - 1,
        "Ratio"    : f"{ratio}:1",
        "Normal"   : n_non,
        "Distress" : n_dist,
        "Train Acc": round(tr_acc, 4),
        "Train F1" : round(tr_f1,  4),
        "Train AUC": round(tr_auc, 4),
        "Test Acc" : round(te_acc, 4),
        "Test F1"  : round(te_f1,  4),
        "Test AUC" : round(te_auc, 4),
    })
    print(f"  {ratio}:1  Normal:{n_non:4d}  "
          f"Tr Acc:{tr_acc:.4f}  "
          f"Te Acc:{te_acc:.4f}  Te F1:{te_f1:.4f}  Te AUC:{te_auc:.4f}")

df_cart = print_table(cart_results, "Experiment 3 — CART Bootstrapping (Altman Z*)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — ANN STANDALONE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("ANN STANDALONE: Feedforward Neural Network")
print("  Architecture : 3 hidden layers (64 → 32 → 16 units), ReLU, Adam")
print("  Batch size   : 35  |  Max epochs: 100  |  Early stopping: patience=10")
print("  Note: Paper used 5 units per layer on 464 firms. Larger layers used")
print("        here because training set is ~10x larger after SMOTE.")
print("=" * 75)

ann = MLPClassifier(
    hidden_layer_sizes  = (64, 32, 16),     # 3 hidden layers
    activation          = "relu",
    solver              = "adam",
    alpha               = 0.0001,           # L2 regularisation
    batch_size          = 35,               # matches paper
    learning_rate_init  = 0.001,            # matches paper
    max_iter            = 100,              # matches paper
    early_stopping      = True,
    validation_fraction = 0.1,              # hold 10% of train for early stop
    n_iter_no_change    = 10,               # patience
    shuffle             = True,
    random_state        = 42,
    verbose             = False,
)

ann.fit(X_res, y_res)

# Training metrics (on full SMOTE-resampled set)
y_tr_pred_ann  = ann.predict(X_res)
y_tr_prob_ann  = ann.predict_proba(X_res)[:, 1]
tr_acc_ann     = accuracy_score(y_res, y_tr_pred_ann)
tr_prec_ann    = precision_score(y_res, y_tr_pred_ann, zero_division=0)
tr_rec_ann     = recall_score(y_res, y_tr_pred_ann, zero_division=0)
tr_f1_ann      = f1_score(y_res, y_tr_pred_ann, zero_division=0)

# Test metrics (original unmodified test set)
y_te_pred_ann  = ann.predict(X_test_arr)
y_te_prob_ann  = ann.predict_proba(X_test_arr)[:, 1]
te_acc_ann     = accuracy_score(y_test_arr, y_te_pred_ann)
te_prec_ann    = precision_score(y_test_arr, y_te_pred_ann, zero_division=0)
te_rec_ann     = recall_score(y_test_arr, y_te_pred_ann, zero_division=0)
te_f1_ann      = f1_score(y_test_arr, y_te_pred_ann, zero_division=0)
te_auc_ann     = roc_auc_score(y_test_arr, y_te_prob_ann)

print(f"\n  Epochs trained     : {ann.n_iter_}")
print(f"  Final loss         : {ann.loss_:.6f}")
print(f"\n  ┌──────────────────────────────────────────────────┐")
print(f"  │              TRAIN SET       TEST SET            │")
print(f"  ├──────────────────────────────────────────────────┤")
print(f"  │  Accuracy  :   {tr_acc_ann:.4f}          {te_acc_ann:.4f}         │")
print(f"  │  Precision :   {tr_prec_ann:.4f}          {te_prec_ann:.4f}         │")
print(f"  │  Recall    :   {tr_rec_ann:.4f}          {te_rec_ann:.4f}         │")
print(f"  │  F1-Score  :   {tr_f1_ann:.4f}          {te_f1_ann:.4f}         │")
print(f"  │  AUC       :     -            {te_auc_ann:.4f}         │")
print(f"  └──────────────────────────────────────────────────┘")

print("\n  Confusion Matrix (Test Set):")
cm = confusion_matrix(y_test_arr, y_te_pred_ann)
print(f"    [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
print(f"     [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")

print("\n  Classification Report (Test Set):")
print(classification_report(y_test_arr, y_te_pred_ann,
                             target_names=["Non-Distressed", "Distressed"]))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — FEATURE IMPORTANCE  (RF + XGBoost, Top 10)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("FEATURE IMPORTANCE: Random Forest + XGBoost (Top 10)")
print("=" * 75)

rf_imp  = pd.Series(exp1_models["RF"].feature_importances_,  index=feature_names)
xgb_imp = pd.Series(exp1_models["XGB"].feature_importances_, index=feature_names)

rf_top10  = rf_imp.sort_values(ascending=False).head(10)
xgb_top10 = xgb_imp.sort_values(ascending=False).head(10)

feat_df = pd.DataFrame({
    "Feature"            : feature_names,
    "RF_Importance"      : exp1_models["RF"].feature_importances_,
    "XGBoost_Importance" : exp1_models["XGB"].feature_importances_,
}).sort_values("RF_Importance", ascending=False)

print("\n  Random Forest — Top 10 Predictors (Altman Z* label):")
for feat, imp in rf_top10.items():
    print(f"    {feat:<50s}  {imp:.6f}")

print("\n  XGBoost — Top 10 Predictors (Altman Z* label):")
for feat, imp in xgb_top10.items():
    print(f"    {feat:<50s}  {imp:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE ALL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("SAVING ALL RESULTS")
print("=" * 75)

df_exp1.to_csv("z_results_exp1_predictive.csv",     index=False)
df_exp2.to_csv("z_results_exp2_rfecv.csv",           index=False)
df_cart.to_csv("z_results_exp3_cart_bootstrap.csv",  index=False)
feat_df.to_csv("z_feature_importance.csv",           index=False)

pd.DataFrame(rfecv_summary).to_csv("z_results_rfecv_summary.csv", index=False)

ann_row = pd.DataFrame([{
    "Train Acc" : round(tr_acc_ann,  4),
    "Test Acc"  : round(te_acc_ann,  4),
    "Precision" : round(te_prec_ann, 4),
    "Recall"    : round(te_rec_ann,  4),
    "F1"        : round(te_f1_ann,   4),
    "AUC"       : round(te_auc_ann,  4),
    "Epochs"    : ann.n_iter_,
    "Final Loss": round(ann.loss_, 6),
}])
ann_row.to_csv("z_results_ann.csv", index=False)

print("  z_results_exp1_predictive.csv")
print("  z_results_exp2_rfecv.csv")
print("  z_results_exp3_cart_bootstrap.csv")
print("  z_results_ann.csv")
print("  z_results_rfecv_summary.csv")
print("  z_feature_importance.csv")

print("\n" + "=" * 75)
print("FINAL SUMMARY — ALL MODELS (Altman Z*-Score Label)")
print("=" * 75)

all_results = (
    [dict(r, Experiment="Exp1 Predictive") for r in exp1_results]
    + [dict(r, Experiment="Exp2 RFECV")     for r in exp2_results]
    + [{
        "Experiment": "ANN",
        "Model"     : "ANN (3-layer MLP)",
        "Train Acc" : round(tr_acc_ann,  4),
        "Test Acc"  : round(te_acc_ann,  4),
        "Precision" : round(te_prec_ann, 4),
        "Recall"    : round(te_rec_ann,  4),
        "F1"        : round(te_f1_ann,   4),
        "AUC"       : round(te_auc_ann,  4),
    }]
)

summary_df = pd.DataFrame(all_results)
cols_order = ["Experiment", "Model", "Train Acc", "Test Acc",
              "Precision", "Recall", "F1", "AUC"]
print(summary_df[cols_order].to_string(index=False))
summary_df[cols_order].to_csv("z_results_all_models.csv", index=False)
print("\n  z_results_all_models.csv")

print("\n" + "=" * 75)
print("PIPELINE COMPLETE — ALTMAN Z*-SCORE")
print("=" * 75)
