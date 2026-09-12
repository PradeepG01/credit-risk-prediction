"""
Credit Risk / Loan Default Prediction
--------------------------------------
Predicts whether a loan applicant is likely to default, using a
synthetic dataset shaped like real credit-risk data (income, credit
score, loan amount, employment length, debt-to-income ratio, etc).

To use a REAL dataset instead (recommended before putting this on
GitHub): download "Give Me Some Credit" or "Loan Default Prediction"
from Kaggle, and swap out generate_synthetic_data() for
pd.read_csv('your_dataset.csv').

Pipeline:
1. Load / generate data
2. Explore + clean
3. Feature engineering
4. Train/test split
5. Train a Logistic Regression baseline + a Random Forest
6. Evaluate with accuracy, precision, recall, F1, ROC-AUC
7. Inspect feature importance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


def generate_synthetic_data(n_samples=5000, random_state=42):
    """Creates a synthetic loan applicant dataset with realistic relationships:
    lower income, lower credit score, and higher debt-to-income ratio all
    push default probability up. Replace this with a real Kaggle dataset
    for your actual GitHub upload.
    """
    rng = np.random.default_rng(random_state)

    income = rng.normal(50000, 20000, n_samples).clip(10000, 200000)
    credit_score = rng.normal(650, 80, n_samples).clip(300, 850)
    loan_amount = rng.normal(15000, 8000, n_samples).clip(1000, 60000)
    employment_years = rng.exponential(5, n_samples).clip(0, 40)
    debt_to_income = rng.normal(0.35, 0.15, n_samples).clip(0, 1)
    num_credit_lines = rng.integers(1, 15, n_samples)
    age = rng.normal(38, 12, n_samples).clip(21, 75)

    # Build a "risk score" from these features, then convert it into a
    # default probability. This is what makes the synthetic labels realistic
    # rather than random noise.
    risk_score = (
        -0.00003 * income
        - 0.01 * credit_score
        + 0.00005 * loan_amount
        - 0.03 * employment_years
        + 2.5 * debt_to_income
        + 0.05 * num_credit_lines
    )
    default_prob = 1 / (1 + np.exp(-risk_score + 5))  # sigmoid to squash into [0,1]
    defaulted = rng.binomial(1, default_prob)

    df = pd.DataFrame({
        'income': income.round(2),
        'credit_score': credit_score.round(0),
        'loan_amount': loan_amount.round(2),
        'employment_years': employment_years.round(1),
        'debt_to_income': debt_to_income.round(3),
        'num_credit_lines': num_credit_lines,
        'age': age.round(0),
        'defaulted': defaulted,
    })
    return df


def train_and_evaluate(df):
    X = df.drop(columns=['defaulted'])
    y = df['defaulted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling matters for Logistic Regression (distance/gradient-based),
    # not strictly necessary for Random Forest, but doesn't hurt.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    }

    results = {}
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds),
            'f1': f1_score(y_test, preds),
            'roc_auc': roc_auc_score(y_test, probs),
            'confusion_matrix': confusion_matrix(y_test, preds),
        }

        print(f'\n--- {name} ---')
        print(classification_report(y_test, preds, target_names=['No Default', 'Default']))
        print(f"ROC-AUC: {results[name]['roc_auc']:.3f}")

    # Feature importance from the Random Forest - useful to explain in an
    # interview: which factors drive the model's decisions.
    rf_model = models['Random Forest']
    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print('\n--- Feature Importance (Random Forest) ---')
    print(importance)

    return results, models, scaler


if __name__ == '__main__':
    df = generate_synthetic_data()
    print('Dataset shape:', df.shape)
    print('\nDefault rate:', df['defaulted'].mean().round(3))
    print('\nFirst 5 rows:\n', df.head())

    results, models, scaler = train_and_evaluate(df)
