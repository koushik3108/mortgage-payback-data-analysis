# 🏦 Mortgage Payback Analytics  
## Predicting Loan Default and Payoff Behavior Using Machine Learning  

**Author:** Sai Koushik Soma    

---

## 📌 Project Overview

This project builds a predictive framework to classify mortgage loans into:

- **Current**
- **Default**
- **Payoff**

Using borrower characteristics, loan attributes, and macroeconomic indicators, this project helps financial institutions:

- Detect high-risk borrowers early  
- Improve credit-risk assessment  
- Forecast prepayment behavior  
- Strengthen mortgage portfolio stability  

The dataset contains **50,000 borrowers observed across 60 periods**, and modeling is performed using each borrower’s final loan outcome.

---

## 🎯 Business Problem

Mortgage lenders face two major risks:

- ❌ **Default Risk** → Financial loss and liquidity issues  
- ❌ **Prepayment Risk** → Loss of expected interest income  

Traditional static credit scoring fails to capture:

- Economic fluctuations  
- Changing loan leverage  
- Property value dynamics  
- Borrower aging effects  

This project provides a **data-driven, machine-learning approach** to predict final loan outcomes and segment borrower risk.

---

## 🎯 Business Goals

✔ Improve default detection  
✔ Enhance prepayment forecasting  
✔ Reduce credit losses  
✔ Strengthen underwriting fairness  
✔ Enable borrower segmentation for targeted strategies  

---

## 📊 Dataset Summary

- **50,000 borrowers**
- 60 time periods
- Panel structure collapsed to final borrower snapshot

### Key Predictors

**Borrower & Loan Features**
- FICO_orig_time
- LTV_time
- interest_rate_time
- loan_age (engineered)
- rate_delta (engineered)

**Macroeconomic Indicators**
- hpi_time
- gdp_time
- uer_time

**Property & Structural Variables**
- REtype indicators
- investor_orig_time

**Target Variable**
- `status_lbl`
  - 0 = Current  
  - 1 = Default  
  - 2 = Payoff  

---

## ⚙️ Data Preprocessing

✔ Logical validation of financial variables  
✔ Range checks for LTV, FICO, interest rates  
✔ Outlier detection  
✔ KNN imputation (Gower distance)  
✔ Feature engineering:
  - `loan_age`
  - `rate_delta`
  - `hpi_change`

✔ 70/30 Stratified Train-Test Split  
✔ Training set balanced via upsampling  
✔ 5-fold CV used in LASSO  

---

## 🤖 Models Implemented

- Multinomial Logistic Regression  
- LASSO Regularized Logistic Regression  
- Random Forest  
- XGBoost  
- K-Means Clustering (Unsupervised)

---

## 📈 Model Performance (Test Set)

| Model | Accuracy | Default Recall | AUC (Default vs Rest) |
|--------|----------|---------------|-----------------------|
| Logistic Regression | 0.779 | 0.719 | 0.862 |
| LASSO | 0.779 | 0.719 | 0.862 |
| Random Forest | 0.818 | 0.706 | 0.888 |
| **XGBoost** | **0.816** | **0.768** | **0.894** |

### 🏆 Final Model Selected: XGBoost

- Highest Default Recall
- Highest AUC
- Strongest risk ranking power
- Economically coherent predictions

---

## 🔍 Key Risk Drivers Identified

Across models and clustering:

- 📉 High LTV → Higher default probability  
- 📉 Rising unemployment → Increased risk  
- 📈 Positive HPI growth → Higher payoff likelihood  
- 📈 High FICO → Stronger loan stability  
- 🔁 Rate differentials → Refinancing behavior  

---

## 🧩 Borrower Segmentation (K-Means)

Three distinct borrower clusters:

| Cluster | Profile | Behavior |
|----------|---------|----------|
| Cluster 1 | Low LTV, Positive HPI | High Payoff |
| Cluster 2 | High LTV, Weak Economy | High Default |
| Cluster 3 | High FICO, Seasoned Loans | Mostly Current |

Segmentation aligns strongly with supervised model outputs.

---

## 📊 Evaluation Metrics Used

Accuracy

Precision

Recall (Default priority)

F1 Score

ROC-AUC

Default recall was prioritized due to its financial impact.

## 🚀 Business Impact

This framework enables lenders to:

Identify high-risk borrowers early

Improve capital allocation

Forecast prepayments more accurately

Segment portfolios by behavioral risk

Support regulatory transparency

## 🏁 Conclusion

This project demonstrates how:

Structured data preprocessing

Feature engineering

Class balancing

Ensemble machine learning

Borrower segmentation

can transform raw mortgage panel data into a robust, actionable credit-risk system.

XGBoost delivers the strongest predictive performance, while Logistic Regression provides regulatory transparency. Together, they create a balanced, business-ready solution for mortgage risk management.

⭐ If you found this project valuable, feel free to star the repository.
