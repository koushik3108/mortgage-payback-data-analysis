# Mortgage Payback Analytics 
Predicting Loan Default and Payoff Behavior Using Machine Learning

## Overview
Mortgage lending carries two major risks:
- **Default risk** → credit losses, higher servicing cost, liquidity stress
- **Early payoff risk** → loss of expected interest income, cashflow forecasting uncertainty

This project builds a data-driven framework to classify mortgage outcomes into:
- **Current**
- **Default**
- **Payoff**

It also performs **borrower segmentation** using clustering to identify groups with similar repayment behavior.

## Dataset
- ~**50,000** borrowers
- **60 time periods** (panel/long format)
- Includes:
  - Borrower and loan attributes (e.g., **FICO**, **LTV**, **interest rates**, balances)
  - Macroeconomic indicators (e.g., **HPI**, **GDP growth**, **unemployment rate**)
- Important note: the modeling dataset is created by **collapsing each borrower to their last observed record**, so the prediction aligns with the final loan outcome lenders care about (**Current / Default / Payoff**).

## Problem Statement
Traditional credit scoring (static FICO-based rules) may miss dynamic risk drivers like:
- rate changes
- unemployment changes
- house price movement

This project predicts final loan behavior and explains the main drivers of each outcome.

## Goals
### Business Goals
- Identify high-risk borrowers early for proactive intervention
- Improve payoff forecasting to stabilize revenue expectations
- Enhance portfolio risk segmentation and decision-making

### Analytics Goals
- Train multiclass ML models to predict **Current / Default / Payoff**
- Handle class imbalance using **class balancing (upsampling)**
- Evaluate models using **Accuracy, Precision, Recall, F1, ROC-AUC**
- Segment borrowers using **K-Means clustering** for behavioral profiling

## Approach
### 1) Data Preparation
- Data cleaning + logical consistency checks
- Missing values handled with **KNN Imputation**
- Feature engineering examples:
  - `loan_age`
  - `rate_delta` (current rate vs origination rate)
  - `hpi_change`

### 2) Feature Diagnostics
- Multicollinearity check using **VIF**
- Redundant/high-collinearity features removed (e.g., overlapping balance fields, origination vs time-based duplicates)

### 3) Train/Test Strategy
- **70/30 stratified split**
- Training set balanced via **upsampling** to reduce bias toward the majority class (Current)

### 4) Models
Supervised multiclass classification:
- Multinomial Logistic Regression (baseline + interpretability)
- LASSO Multinomial Logistic Regression (feature selection via CV)
- Random Forest (nonlinear patterns + importance)
- XGBoost (high accuracy + strong ranking power)

Unsupervised segmentation:
- K-Means clustering to identify borrower groups with distinct risk behavior

## Results Summary (Test Set)
| Model | Accuracy | Default Recall | AUC (Default vs Non-Default) |
|------|----------|----------------|------------------------------|
| Multinomial Logistic Regression | ~0.7795 | ~0.7198 | ~0.862 |
| LASSO | ~0.7790 | ~0.7191 | ~0.862 |
| Random Forest | ~0.8183 | ~0.7061 | ~0.888 |
| XGBoost | ~0.8162 | **~0.7681** | **~0.894** |

**Key takeaway:** XGBoost delivered the best overall balance for default detection and risk ranking, while logistic models remain useful for explainability.

## Clustering Insights (K-Means)
Three borrower segments were identified with distinct profiles:
- **Payoff-prone group**: lower LTV + favorable HPI change → high payoff propensity
- **High-risk group**: very high LTV + negative HPI change + higher unemployment → high default rates
- **Stable/current group**: higher FICO + seasoned loans + stable macro → mostly current

This segmentation complements supervised models by explaining *why* groups behave differently.

## Repository Structure (Suggested)
You can organize your repo like this:
