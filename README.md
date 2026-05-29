# Beyond the Red Flags: Real-Time Financial Distress Forecasting for NSE-Listed Firms

## Overview

This project develops a comprehensive machine learning and survival analysis framework for predicting financial distress among companies listed on the National Stock Exchange (NSE) of India.

Using historical financial ratio data from 1,797 NSE-listed firms spanning 2012–2022, the study identifies distressed firms through the Revised Altman Z*-Score and applies advanced machine learning techniques to forecast financial distress while also examining the timing and risk factors associated with distress events.

---

## Objectives

* Identify financially distressed and healthy firms using the Revised Altman Z*-Score (1983).
* Build predictive machine learning models for financial distress forecasting.
* Address class imbalance using SMOTE + ENN.
* Perform feature selection using RFECV.
* Compare the performance of multiple classification algorithms.
* Analyze time-to-distress using Survival Analysis techniques.
* Identify key financial indicators influencing distress risk.

---

## Dataset

### Source

National Stock Exchange (NSE), India

### Coverage

* Time Period: 2012–2022
* Unique Firms: 1,797
* Firm-Year Observations: 19,767
* Financial Variables: 98

### Final Processed Dataset

* Labelled Observations: 15,513
* Distressed Firms: 633
* Healthy Firms: 14,880
* Distress Rate: 4.1%
* Final Features Used: 46

---

## Financial Distress Labelling

The project uses the **Revised Altman Z*-Score (1983)** designed for non-manufacturing and emerging-market firms.

### Formula

Z* = 3.25 + 6.56X₁ + 3.26X₂ + 6.72X₃ + 1.05X₄

Where:

* X₁ = Working Capital / Total Assets
* X₂ = Retained Earnings / Total Assets
* X₃ = EBIT / Total Assets
* X₄ = Market Value Equity / Book Liabilities

### Classification

| Z*-Score    | Status               |
| ----------- | -------------------- |
| < 1.10      | Distressed           |
| 1.10 – 2.60 | Grey Zone (Excluded) |
| > 2.60      | Healthy              |

---

## Machine Learning Pipeline

### Data Preparation

* Missing Value Handling using KNN Imputation
* Outlier Treatment using Z-Score Filtering
* Multicollinearity Removal
* Data Leakage Prevention through removal of Z*-Score construction variables
* Min-Max Feature Scaling

### Class Imbalance Handling

SMOTE + ENN

### Feature Selection

Recursive Feature Elimination with Cross Validation (RFECV)

---

## Models Evaluated

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* XGBoost
* Artificial Neural Network (ANN)

---

## Experimental Design

### Experiment 1

Full feature set + SMOTE + ENN

### Experiment 2

RFECV-selected features + SMOTE + ENN

### Experiment 3

CART Bootstrapping under varying imbalance ratios (1:1 to 10:1)

### ANN Baseline

Three-layer feedforward neural network:

64 → 32 → 16 neurons

---

## Results

### Experiment 1

| Model               | Accuracy | F1 Score | AUC  |
| ------------------- | -------- | -------- | ---- |
| Logistic Regression | 0.81     | 0.26     | 0.88 |
| Decision Tree       | 0.96     | 0.66     | 0.91 |
| Random Forest       | 0.90     | 0.44     | 0.97 |
| SVM                 | 0.84     | 0.32     | 0.90 |
| XGBoost             | 0.97     | 0.70     | 0.99 |

### Best Performing Model

🏆 **XGBoost**

* Accuracy: 96.85%
* Recall: 93%
* F1 Score: 0.70
* AUC: 0.99

---

## Survival Analysis

### Techniques Used

* Kaplan-Meier Survival Estimation
* Cox Proportional Hazards Model

### Survival Dataset

* Firms: 1,604
* Distress Events: 153
* Observation Window: Up to 10 Years

### Key Findings

* Approximately 89% of firms remained financially healthy over a 10-year period.
* Firms in the lowest Z*-Score quartile exhibited a 19.7% distress event rate.
* Firms in the highest quartile exhibited only a 1.2% distress event rate.

---

## Cox Proportional Hazards Results

### Significant Predictors

#### Total Debt to Capitalization

* Hazard Ratio: 2.354
* p < 0.001

Higher leverage significantly accelerates financial distress.

#### Net Profit Margin

* Hazard Ratio: 0.457
* p = 0.001

Higher profitability significantly reduces distress risk.

### Model Performance

* Concordance Index (C-Index): 0.706

---

## Key Insights

* XGBoost consistently outperformed all benchmark models.
* RFECV reduced dimensionality while maintaining predictive performance.
* Leverage emerged as the strongest risk factor for financial distress.
* Profitability acted as the strongest protective factor.
* Survival analysis provided valuable insights into the timing of distress events beyond simple classification.

---

## Tech Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Imbalanced-Learn
* Lifelines
* Matplotlib
* Seaborn

### Techniques

* SMOTE + ENN
* RFECV
* GridSearchCV
* Kaplan-Meier Estimation
* Cox Proportional Hazards Modeling

---

## Repository Structure

```text
├── altman_z_score_corrected.py
├── zscore_feature_removal_and_split.py
├── survival_analysis.py
├── dashboards.py
├── XG.py
├── README.md
```
This project is intended for academic and research purposes only. The predictions generated should not be considered financial or investment advice.
