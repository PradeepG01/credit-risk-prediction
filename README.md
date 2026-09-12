# Credit Risk / Loan Default Prediction

A machine learning model that predicts whether a loan applicant is likely to
default, based on financial and demographic features. Built with scikit-learn.

## Why I built this

Credit risk scoring is one of the most common real-world applications of
ML in fintech - banks and lenders use models like this to decide loan
approvals and interest rates. It's a strong way to show applied ML skill
for finance-related analyst or ML roles.

## Important: before uploading to GitHub
This script currently generates a **synthetic** dataset so it runs
standalone with no downloads required. For a stronger portfolio piece,
replace `generate_synthetic_data()` with a real dataset - the "Give Me
Some Credit" or "Loan Default Prediction" datasets on Kaggle work well
and use very similar features.

## Tech Stack
- **Python, pandas, NumPy** — data handling
- **scikit-learn** — Logistic Regression + Random Forest models
- **matplotlib / seaborn** — visualization (add plots for the GitHub version)

## Approach
1. Load/generate applicant data (income, credit score, loan amount,
   employment length, debt-to-income ratio, number of credit lines, age)
2. Train/test split (80/20, stratified so the default rate is consistent
   in both sets)
3. Feature scaling for Logistic Regression
4. Train two models — a simple, interpretable baseline (Logistic
   Regression) and a stronger model (Random Forest) — and compare them
5. Evaluate with accuracy, precision, recall, F1, and ROC-AUC (accuracy
   alone is misleading here since defaults are usually a minority class)
6. Inspect which features the Random Forest relies on most

## Running It
```
pip install -r requirements.txt
python credit_risk_model.py
```

## What I'd add next
- Swap in a real Kaggle dataset
- Handle class imbalance explicitly (SMOTE or class weighting)
- Add SHAP values for per-prediction explainability - important in
  finance, where a rejected applicant may need to know why
- Deploy as a simple API endpoint (Flask/FastAPI) that takes applicant
  details and returns a risk score
