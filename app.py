#!/usr/bin/env python3
"""
Telco Customer Churn Dashboard — Streamlit Application
Analyzes customer churn patterns with ML predictions and AI assistant.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import duckdb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Telco Churn Dashboard",
    page_icon="📊",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

DATA_PATH = Path("data") / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TENURE_BINS   = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49-72m"]

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

AUTO_PAYMENT_METHODS = ["Bank transfer (automatic)", "Credit card (automatic)"]

FEATURE_COLS = [
    "Contract", "PaymentMethod", "InternetService", "tenure_bucket",
    "tenure", "MonthlyCharges", "TotalCharges", "services_count",
    "is_month_to_month", "paperless_flag", "auto_payment_flag", "senior_citizen_flag",
]
TARGET_COL          = "churn_flag"
CATEGORICAL_FEATURES = ["Contract", "PaymentMethod", "InternetService", "tenure_bucket"]
NUMERIC_FEATURES     = [
    "tenure", "MonthlyCharges", "TotalCharges", "services_count",
    "is_month_to_month", "paperless_flag", "auto_payment_flag", "senior_citizen_flag",
]

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
N_ESTIMATORS  = 200
MAX_ITER      = 1000

RISK_THRESHOLDS = {"low": 0.30, "medium": 0.60, "high": 0.80}

FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

CUSTOMERS_SCHEMA = """
Table: customers_clean
Columns: customerID (VARCHAR), gender (VARCHAR), SeniorCitizen (VARCHAR Yes/No),
  Partner (VARCHAR), Dependents (VARCHAR), tenure (INT), PhoneService (VARCHAR),
  MultipleLines (VARCHAR), InternetService (VARCHAR), OnlineSecurity (VARCHAR),
  OnlineBackup (VARCHAR), DeviceProtection (VARCHAR), TechSupport (VARCHAR),
  StreamingTV (VARCHAR), StreamingMovies (VARCHAR), Contract (VARCHAR),
  PaperlessBilling (VARCHAR), PaymentMethod (VARCHAR), MonthlyCharges (FLOAT),
  TotalCharges (FLOAT), Churn (VARCHAR), tenure_bucket (VARCHAR),
  services_count (INT), is_month_to_month (INT), paperless_flag (INT),
  auto_payment_flag (INT), churn_flag (INT)
DuckDB dialect. Use only SELECT. churn_flag = 1 means churned.
""".strip()

PREDEFINED_QUERIES = {
    "contract": {
        "sql": """SELECT Contract, COUNT(*) AS customers,
                         ROUND(AVG(churn_flag)*100, 2) AS churn_rate_pct
                  FROM customers_clean
                  GROUP BY Contract ORDER BY churn_rate_pct DESC""",
        "explanation": (
            "Month-to-month contracts show the highest churn at 42.71%, nearly 15× higher than "
            "two-year contracts (2.83%). Migrating customers to annual contracts is the single "
            "most impactful retention lever."
        ),
    },
    "payment": {
        "sql": """SELECT PaymentMethod, COUNT(*) AS customers,
                         ROUND(AVG(churn_flag)*100, 2) AS churn_rate_pct
                  FROM customers_clean
                  GROUP BY PaymentMethod ORDER BY churn_rate_pct DESC""",
        "explanation": (
            "Electronic check users churn at 45.29%, nearly 3× the rate of automatic payment "
            "customers (~15–17%). Incentivising auto-pay enrolment is a high-ROI retention action."
        ),
    },
    "revenue": {
        "sql": """SELECT ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_at_risk,
                         ROUND(SUM(MonthlyCharges)*12, 2) AS annual_revenue_at_risk,
                         COUNT(*) AS churned_customers
                  FROM customers_clean WHERE churn_flag = 1""",
        "explanation": (
            "Churning customers represent $139,131/month ($1.67M/year) in recurring revenue at risk. "
            "A 5-point reduction in churn rate would protect over $300K annually."
        ),
    },
    "tenure": {
        "sql": """SELECT tenure_bucket,
                         ROUND(AVG(tenure), 1) AS avg_tenure_months,
                         COUNT(*) AS customers,
                         ROUND(AVG(churn_flag)*100, 2) AS churn_rate_pct
                  FROM customers_clean
                  GROUP BY tenure_bucket ORDER BY avg_tenure_months""",
        "explanation": (
            "New customers in their first year show 47.44% churn, confirming the first 12 months "
            "as the critical retention window. Structured onboarding programmes for this cohort "
            "yield the highest ROI."
        ),
    },
    "monthly": {
        "sql": """SELECT CASE WHEN churn_flag=1 THEN \'Churned\' ELSE \'Retained\' END AS status,
                         ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
                         COUNT(*) AS customers
                  FROM customers_clean
                  GROUP BY churn_flag""",
        "explanation": (
            "Churned customers pay on average $74.44/month vs $61.27 for retained customers, "
            "indicating that higher-value customers are disproportionately at risk and warrant "
            "premium retention investment."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CACHING: LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean_data():
    """Load CSV and apply the same cleaning logic as the notebook."""

    if not DATA_PATH.exists():
        st.error(
            f"""Dataset not found at '{DATA_PATH}'.

Please place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' inside the 'data' folder.

Expected project structure:

Telco-Customer-Churn-Dashboard/
├── app.py
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
        )
        st.stop()

    # <-- This line must be OUTSIDE the if block
    df = pd.read_csv(DATA_PATH)

    # Continue with the cleaning...
    # Recode SeniorCitizen 0/1 → "No"/"Yes"
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Coerce TotalCharges to numeric (11 blanks → NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Tenure bucket
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True
    )
    df["tenure_bucket"] = df["tenure_bucket"].astype(str)

    # Services count
    df["services_count"] = (
        df[SERVICE_COLS].isin(["Yes", "DSL", "Fiber optic"]).sum(axis=1)
    )

    # Binary flags
    df["is_month_to_month"]   = (df["Contract"] == "Month-to-month").astype(int)
    df["paperless_flag"]      = (df["PaperlessBilling"] == "Yes").astype(int)
    df["auto_payment_flag"]   = df["PaymentMethod"].isin(AUTO_PAYMENT_METHODS).astype(int)
    df["churn_flag"]          = (df["Churn"] == "Yes").astype(int)
    df["senior_citizen_flag"] = (df["SeniorCitizen"] == "Yes").astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CACHING: SQL QUERIES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_sql_queries(_df):
    """Run all KPI SQL queries — returns dict of result DataFrames."""
    con = duckdb.connect()
    con.register("customers_clean", _df)

    q1 = con.execute("""
        SELECT
            COUNT(*)                              AS total_customers,
            SUM(churn_flag)                       AS churned,
            COUNT(*) - SUM(churn_flag)            AS retained,
            ROUND(AVG(churn_flag)*100, 2)         AS churn_rate_pct
        FROM customers_clean
    """).df()

    q2 = con.execute("""
        SELECT Contract,
               COUNT(*)                     AS customers,
               SUM(churn_flag)              AS churned,
               ROUND(AVG(churn_flag)*100,2) AS churn_rate_pct
        FROM customers_clean
        GROUP BY Contract
        ORDER BY churn_rate_pct DESC
    """).df()

    q3 = con.execute("""
        SELECT PaymentMethod,
               COUNT(*)                     AS customers,
               SUM(churn_flag)              AS churned,
               ROUND(AVG(churn_flag)*100,2) AS churn_rate_pct
        FROM customers_clean
        GROUP BY PaymentMethod
        ORDER BY churn_rate_pct DESC
    """).df()

    q4 = con.execute("""
        SELECT InternetService,
               COUNT(*)                     AS customers,
               SUM(churn_flag)              AS churned,
               ROUND(AVG(churn_flag)*100,2) AS churn_rate_pct
        FROM customers_clean
        GROUP BY InternetService
        ORDER BY churn_rate_pct DESC
    """).df()

    q5 = con.execute("""
        SELECT tenure_bucket,
               COUNT(*)                     AS customers,
               SUM(churn_flag)              AS churned,
               ROUND(AVG(churn_flag)*100,2) AS churn_rate_pct
        FROM customers_clean
        GROUP BY tenure_bucket
        ORDER BY MIN(tenure)
    """).df()

    q6 = con.execute("""
        SELECT
            ROUND(SUM(MonthlyCharges), 2)    AS monthly_revenue_at_risk,
            ROUND(SUM(MonthlyCharges)*12, 2) AS annual_revenue_at_risk,
            COUNT(*)                         AS churned_customers
        FROM customers_clean
        WHERE churn_flag = 1
    """).df()

    q7 = con.execute("""
        SELECT
            CASE WHEN churn_flag=1 THEN \'Churned\' ELSE \'Retained\' END AS churn_status,
            ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
            ROUND(AVG(TotalCharges),   2) AS avg_total_charges,
            ROUND(AVG(tenure),         1) AS avg_tenure_months,
            COUNT(*)                      AS customers
        FROM customers_clean
        GROUP BY churn_flag
        ORDER BY churn_flag DESC
    """).df()

    q8 = con.execute("""
        SELECT
            CASE WHEN churn_flag=1 THEN \'Churned\' ELSE \'Retained\' END AS churn_status,
            ROUND(MIN(MonthlyCharges), 2) AS min_monthly,
            ROUND(MAX(MonthlyCharges), 2) AS max_monthly,
            ROUND(AVG(MonthlyCharges), 2) AS avg_monthly,
            ROUND(STDDEV(MonthlyCharges), 2) AS std_monthly
        FROM customers_clean
        GROUP BY churn_flag
        ORDER BY churn_flag DESC
    """).df()

    q9 = con.execute("""
        SELECT Contract, PaymentMethod, InternetService,
               COUNT(*)                     AS customers,
               SUM(churn_flag)              AS churned,
               ROUND(AVG(churn_flag)*100,2) AS churn_rate_pct,
               ROUND(AVG(MonthlyCharges),2) AS avg_monthly_charges,
               ROUND(SUM(MonthlyCharges),2) AS total_monthly_revenue
        FROM customers_clean
        GROUP BY Contract, PaymentMethod, InternetService
        HAVING COUNT(*) >= 30
        ORDER BY churn_rate_pct DESC
        LIMIT 10
    """).df()

    con.close()

    return {
        "q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5,
        "q6": q6, "q7": q7, "q8": q8, "q9": q9,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CACHING: TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    """Train LR and RF models — cached so they are trained once."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, roc_curve,
    )

    # Drop rows with NaN in FEATURE_COLS or TARGET_COL
    df_model = _df[FEATURE_COLS + [TARGET_COL]].dropna()
    X = df_model[FEATURE_COLS]
    y = df_model[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer([
        ("cat", ohe, CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])

    lr_pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)),
    ])
    rf_pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)),
    ])

    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    def compute_metrics(pipeline, X_t, y_t):
        y_pred = pipeline.predict(X_t)
        y_prob = pipeline.predict_proba(X_t)[:, 1]
        return {
            "Accuracy":       round(accuracy_score(y_t, y_pred), 4),
            "Precision (wt)": round(precision_score(y_t, y_pred, average="weighted"), 4),
            "Recall (wt)":    round(recall_score(y_t, y_pred, average="weighted"), 4),
            "F1 (wt)":        round(f1_score(y_t, y_pred, average="weighted"), 4),
            "ROC-AUC":        round(roc_auc_score(y_t, y_prob), 4),
        }

    lr_metrics = compute_metrics(lr_pipeline, X_test, y_test)
    rf_metrics = compute_metrics(rf_pipeline, X_test, y_test)

    # Feature names after OHE
    fitted_preprocessor = rf_pipeline.named_steps["pre"]
    ohe_names = list(
        fitted_preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    feature_names = ohe_names + NUMERIC_FEATURES

    # RF feature importance
    rf_importances = rf_pipeline.named_steps["clf"].feature_importances_
    rf_imp_df = pd.DataFrame({"feature": feature_names, "importance": rf_importances})
    rf_imp_df = rf_imp_df.sort_values("importance", ascending=False).head(10)

    # LR coefficients
    lr_coefs = lr_pipeline.named_steps["clf"].coef_[0]
    lr_coef_df = pd.DataFrame({"feature": feature_names, "coefficient": lr_coefs})
    lr_coef_df["abs_coef"] = lr_coef_df["coefficient"].abs()
    lr_coef_df = lr_coef_df.sort_values("abs_coef", ascending=False).head(15)

    # Confusion matrices
    cm_lr = confusion_matrix(y_test, lr_pipeline.predict(X_test))
    cm_rf = confusion_matrix(y_test, rf_pipeline.predict(X_test))

    # ROC curves
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    # Metrics comparison DataFrame
    metrics_df = pd.DataFrame({
        "Model":           ["Logistic Regression", "Random Forest"],
        "Accuracy":        [lr_metrics["Accuracy"],       rf_metrics["Accuracy"]],
        "Precision (wt)":  [lr_metrics["Precision (wt)"], rf_metrics["Precision (wt)"]],
        "Recall (wt)":     [lr_metrics["Recall (wt)"],    rf_metrics["Recall (wt)"]],
        "F1 (wt)":         [lr_metrics["F1 (wt)"],        rf_metrics["F1 (wt)"]],
        "ROC-AUC":         [lr_metrics["ROC-AUC"],        rf_metrics["ROC-AUC"]],
    })

    return {
        "lr_pipeline":   lr_pipeline,
        "rf_pipeline":   rf_pipeline,
        "lr_metrics":    lr_metrics,
        "rf_metrics":    rf_metrics,
        "cm_lr":         cm_lr,
        "cm_rf":         cm_rf,
        "fpr_lr":        fpr_lr,
        "tpr_lr":        tpr_lr,
        "auc_lr":        round(float(auc_lr), 4),
        "fpr_rf":        fpr_rf,
        "tpr_rf":        tpr_rf,
        "auc_rf":        round(float(auc_rf), 4),
        "rf_imp_df":     rf_imp_df,
        "lr_coef_df":    lr_coef_df,
        "X_test":        X_test,
        "y_test":        y_test,
        "feature_names": feature_names,
        "metrics_df":    metrics_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI ASSISTANT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _get_duckdb_con(df):
    """Fresh DuckDB connection with df registered as customers_clean."""
    con = duckdb.connect()
    con.register("customers_clean", df)
    return con


def _validate_sql(sql: str) -> None:
    if FORBIDDEN_SQL_PATTERN.search(sql):
        raise ValueError(f"Forbidden SQL statement detected: {sql[:200]}")


def summarize_churn_drivers(kpi_tables: dict, gemini_model=None) -> dict:
    """
    Summarize the top churn drivers and business recommendations from KPI tables.
    Returns a dict with keys: executive_summary, churn_drivers, recommendations, provider.
    """
    q1 = kpi_tables.get("q1_result", pd.DataFrame())
    q2 = kpi_tables.get("q2_result", pd.DataFrame())
    q3 = kpi_tables.get("q3_result", pd.DataFrame())
    q4 = kpi_tables.get("q4_result", pd.DataFrame())
    q6 = kpi_tables.get("q6_result", pd.DataFrame())

    overall_churn_rate   = q1["churn_rate_pct"].iloc[0]        if not q1.empty else 26.54
    monthly_rev_at_risk  = q6["monthly_revenue_at_risk"].iloc[0] if ("monthly_revenue_at_risk" in q6.columns and not q6.empty) else 139130.0
    annual_rev_at_risk   = q6["annual_revenue_at_risk"].iloc[0]  if ("annual_revenue_at_risk"  in q6.columns and not q6.empty) else monthly_rev_at_risk * 12
    highest_risk_contract = q2.loc[q2["churn_rate_pct"].idxmax(), "Contract"]       if not q2.empty else "Month-to-month"
    highest_risk_payment  = q3.loc[q3["churn_rate_pct"].idxmax(), "PaymentMethod"]  if not q3.empty else "Electronic check"
    highest_risk_internet = q4.loc[q4["churn_rate_pct"].idxmax(), "InternetService"] if not q4.empty else "Fiber optic"

    result = {}
    used_gemini = False

    if gemini_model is not None:
        try:
            prompt = f"""You are a senior telecoms data analyst. Analyse these churn KPIs and write a
structured Markdown report for a C-suite audience.

Key Metrics:
- Overall churn rate: {overall_churn_rate:.2f}%
- Monthly revenue at risk: ${monthly_rev_at_risk:,.0f}
- Annual revenue at risk: ${annual_rev_at_risk:,.0f}
- Highest-risk contract type: {highest_risk_contract}
- Highest-risk payment method: {highest_risk_payment}
- Highest-risk internet service: {highest_risk_internet}

Return ONLY Markdown with these exact sections:
## Executive Summary
## Top 3 Churn Drivers
## Highest Risk Customer Segments
## Revenue at Risk
## Business Recommendations"""
            response = gemini_model.generate_content(prompt)
            md = response.text
            result = {
                "executive_summary": md,
                "churn_drivers":     md,
                "recommendations":   md,
                "provider":          "gemini",
            }
            used_gemini = True
        except Exception:
            pass

    if not used_gemini:
        exec_summary = (
            f"The telco's overall churn rate stands at **{overall_churn_rate:.2f}%**, "
            f"translating to **${monthly_rev_at_risk:,.0f}/month** (${annual_rev_at_risk:,.0f}/year) "
            f"in recurring revenue at risk. The three highest-risk segments are customers on "
            f"**{highest_risk_contract}** contracts, those paying via **{highest_risk_payment}**, "
            f"and **{highest_risk_internet}** internet subscribers."
        )
        churn_drivers = [
            {
                "driver":    f"{highest_risk_contract} contracts",
                "churn_rate": "42.71%",
                "benchmark": "2.83% (Two-year)",
                "impact":    "15.1× uplift vs two-year contracts — largest single driver",
            },
            {
                "driver":    f"{highest_risk_payment} payments",
                "churn_rate": "45.29%",
                "benchmark": "~16% (auto-pay)",
                "impact":    "~3× uplift vs auto-pay; auto-enrolment is a high-ROI lever",
            },
            {
                "driver":    f"{highest_risk_internet} internet",
                "churn_rate": "41.89%",
                "benchmark": "7.40% (No internet)",
                "impact":    "Quality/price perception gap driving premium-segment attrition",
            },
        ]
        recommendations = [
            f"**Contract migration**: incentivise {highest_risk_contract} customers with a 10% discount on 12-month contracts — addresses 42.71% churn in the largest at-risk cohort.",
            f"**Auto-pay enrolment**: offer a $5/month bill credit to {highest_risk_payment} users switching to automatic payments — targets the 45.29% churn segment.",
            f"**Onboarding programme**: launch 90-day structured onboarding for new customers (0–12m tenure has 47.44% churn) — protecting ~$83K/yr with a 5-point churn reduction.",
            f"**Fibre quality initiative**: invest in network quality and proactive support for {highest_risk_internet} subscribers at 41.89% churn.",
            f"**Predictive intervention**: deploy the trained RF model to score all active customers monthly; prioritise proactive outreach for the top-quartile risk cohort.",
        ]
        result = {
            "executive_summary": exec_summary,
            "churn_drivers":     churn_drivers,
            "recommendations":   recommendations,
            "provider":          "rule-based",
        }

    return result


def ask_churn_question(question: str, df: pd.DataFrame, gemini_model=None) -> dict:
    """
    Generate and execute a SQL query for the given business question.
    Returns dict: question, sql, results (DataFrame), explanation.
    """
    question_lower = question.lower()
    sql = None
    explanation = ""
    used_gemini = False

    if gemini_model is not None:
        try:
            prompt = f"""You are a DuckDB SQL expert for a telecom churn analytics database.
Schema:
{CUSTOMERS_SCHEMA}

Question: {question}

Return ONLY a single valid DuckDB SELECT SQL statement. No explanation, no markdown fences.
The query must reference the table 'customers_clean'.
Do not include INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, EXEC, or EXECUTE."""
            response = gemini_model.generate_content(prompt)
            candidate_sql = response.text.strip().strip("```sql").strip("```").strip()
            _validate_sql(candidate_sql)
            con = _get_duckdb_con(df)
            results_df = con.execute(candidate_sql).df()
            con.close()
            explanation = "SQL generated and executed using Google Gemini AI."
            sql = candidate_sql
            used_gemini = True
            return {"question": question, "sql": sql, "results": results_df, "explanation": explanation}
        except Exception:
            pass

    # Rule-based keyword matching
    if any(kw in question_lower for kw in ["contract", "month-to-month", "annual", "two-year"]):
        key = "contract"
    elif any(kw in question_lower for kw in ["payment", "check", "auto", "credit card", "bank"]):
        key = "payment"
    elif any(kw in question_lower for kw in ["revenue", "risk", "money", "charges", "cost"]):
        key = "revenue"
    elif any(kw in question_lower for kw in ["tenure", "new customer", "long", "month"]):
        key = "tenure"
    elif any(kw in question_lower for kw in ["monthly", "average", "charge", "bill"]):
        key = "monthly"
    else:
        key = "contract"  # default

    entry = PREDEFINED_QUERIES[key]
    sql = entry["sql"]
    explanation = entry["explanation"]

    con = _get_duckdb_con(df)
    results_df = con.execute(sql).df()
    con.close()

    return {"question": question, "sql": sql, "results": results_df, "explanation": explanation}


def explain_customer_risk(customer_id: str, df: pd.DataFrame, rf_pipeline, feature_cols: list) -> dict:
    """
    Look up customer, score with RF pipeline, return risk assessment dict.
    """
    customer_df = df[df["customerID"] == customer_id.strip()]
    if customer_df.empty:
        return {
            "customer_id":            customer_id,
            "churn_probability":       None,
            "risk_level":              "Unknown",
            "profile_summary":         "Customer ID not found in the dataset.",
            "key_risk_factors":        [],
            "retention_recommendation": "Verify the customer ID and try again.",
            "marketing_offer":         "N/A",
            "provider":                "rule-based",
        }

    row = customer_df.iloc[0]

    # Derive senior_citizen_flag if missing from feature set
    customer_features = customer_df[feature_cols].copy()

    proba = rf_pipeline.predict_proba(customer_features)[0, 1]
    churn_prob_pct = round(float(proba) * 100, 1)

    if proba < RISK_THRESHOLDS["low"]:
        risk_level = "Low"
        risk_emoji = "🟢"
    elif proba < RISK_THRESHOLDS["medium"]:
        risk_level = "Medium"
        risk_emoji = "🟡"
    elif proba < RISK_THRESHOLDS["high"]:
        risk_level = "High"
        risk_emoji = "🔴"
    else:
        risk_level = "Critical"
        risk_emoji = "🚨"

    profile = (
        f"{row.get('Contract', 'N/A')} contract, "
        f"{row.get('InternetService', 'N/A')} internet, "
        f"{row.get('PaymentMethod', 'N/A')}, "
        f"tenure={int(row.get('tenure', 0))}m, "
        f"${row.get('MonthlyCharges', 0):.2f}/month"
    )

    risk_factors = []
    if str(row.get("Contract", "")) == "Month-to-month":
        risk_factors.append("Month-to-month contract (42.71% segment churn rate)")
    if str(row.get("PaymentMethod", "")) == "Electronic check":
        risk_factors.append("Electronic check payment (45.29% churn rate)")
    if str(row.get("InternetService", "")) == "Fiber optic":
        risk_factors.append("Fiber optic service (41.89% churn rate)")
    if int(row.get("tenure", 99)) <= 12:
        risk_factors.append(f"New customer (tenure={int(row.get('tenure', 0))}m; 0–12m cohort has 47.44% churn)")
    if float(row.get("MonthlyCharges", 0)) > 65:
        risk_factors.append(f"High monthly charges (${row.get('MonthlyCharges', 0):.0f}/month vs $61.27 retained avg)")
    if int(row.get("paperless_flag", 0)) == 1:
        risk_factors.append("Paperless billing (correlated with higher churn)")
    if int(row.get("auto_payment_flag", 0)) == 0:
        risk_factors.append("No automatic payment (higher churn segment)")
    if not risk_factors:
        risk_factors.append("No dominant risk signals identified — model-driven score applies")

    if risk_level in ("High", "Critical"):
        rec = "Immediate outreach: offer contract upgrade incentive + auto-pay bill credit."
        offer = "10% discount on 12-month contract + $5/mo auto-pay credit"
    elif risk_level == "Medium":
        rec = "Proactive engagement: send personalised satisfaction survey and loyalty reward."
        offer = "Loyalty reward: one month free streaming add-on"
    else:
        rec = "Low risk — routine engagement. Monitor quarterly."
        offer = "Standard renewal notice at next billing cycle"

    return {
        "customer_id":            customer_id,
        "churn_probability":       churn_prob_pct,
        "risk_level":              f"{risk_emoji} {risk_level}",
        "profile_summary":         profile,
        "key_risk_factors":        risk_factors,
        "retention_recommendation": rec,
        "marketing_offer":         offer,
        "provider":                "rule-based",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
  [data-testid="metric-container"] {
    background: #F3F7FF;
    border: 1px solid #BBDEFB;
    border-radius: 8px;
    padding: 16px;
  }
  [data-testid="stMetricValue"] { color: #0D47A1; font-size: 1.8rem; font-weight: 700; }
  [data-testid="stMetricLabel"] { color: #1565C0; font-weight: 600; }
  .stTabs [data-baseweb="tab"] { font-weight: 600; }
  .footer { text-align: center; color: #888; font-size: 0.85rem; margin-top: 2rem; padding: 1rem; border-top: 1px solid #E3E8F0; }
</style>
"""

FOOTER_HTML = """
<div class="footer">
  Created by Data Analyst | Telco Customer Churn Analytics Platform | Built with Streamlit &amp; Plotly
</div>
"""

PLOT_FONT  = {"family": "Inter, Arial, sans-serif"}
COLOR_HIGH = "#E53935"
COLOR_LOW  = "#1565C0"
TEMPLATE   = "plotly_white"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: HORIZONTAL BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
def make_hbar_chart(categories, values, title):
    """Return a Plotly horizontal bar figure. Highest bar is red, others blue."""
    max_val = max(values) if values else 0
    colors  = [COLOR_HIGH if v == max_val else COLOR_LOW for v in values]
    labels  = [f"{v:.1f}%" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        template=TEMPLATE,
        font=PLOT_FONT,
        xaxis=dict(title="Churn Rate (%)", showgrid=True),
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=20, r=40, t=50, b=20),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_executive_overview(df_clean, sql_data, ml_data):

    st.markdown("# 🏠 Executive Overview")
    st.markdown("**Real-time churn analytics and business intelligence for telecom leadership.**")

    q1 = sql_data["q1"]
    q6 = sql_data["q6"]
    q7 = sql_data["q7"]

    churn_rate   = float(q1["churn_rate_pct"].iloc[0])
    total_cust   = int(q1["total_customers"].iloc[0])
    churned_n    = int(q1["churned"].iloc[0])
    retained_n   = int(q1["retained"].iloc[0])
    rev_monthly  = float(q6["monthly_revenue_at_risk"].iloc[0])
    rev_annual   = float(q6["annual_revenue_at_risk"].iloc[0])
    avg_charge   = float(df_clean["MonthlyCharges"].mean())

    rf_accuracy  = ml_data["rf_metrics"]["Accuracy"]
    rf_recall    = ml_data["rf_metrics"]["Recall (wt)"]

    churned_q7   = q7[q7["churn_status"] == "Churned"]
    avg_m_churned = (
        float(churned_q7["avg_monthly_charges"].iloc[0])
        if not churned_q7.empty else 74.44
    )

    # ── KPI Row ──────────────────────────────────────────────────────────────
    st.markdown("---")
    cols = st.columns(6)
    with cols[0]:
        st.metric("Total Customers", f"{total_cust:,}")
    with cols[1]:
        st.metric("Churn Rate", f"{churn_rate:.2f}%", delta=f"-{churn_rate:.2f}% target: <20%", delta_color="inverse")
    with cols[2]:
        st.metric("Revenue at Risk", f"${rev_monthly:,.0f}/mo")
    with cols[3]:
        st.metric("Avg Monthly Charges", f"${avg_charge:.2f}")
    with cols[4]:
        st.metric("RF Accuracy", f"{rf_accuracy*100:.1f}%")
    with cols[5]:
        st.metric("RF Recall", f"{rf_recall*100:.1f}%")

    st.markdown("---")

    # ── Donut + Revenue Risk ──────────────────────────────────────────────────
    col_donut, col_rev = st.columns([1, 1])

    with col_donut:
        st.subheader("Churn Distribution")
        fig_donut = go.Figure(go.Pie(
            labels=["Churned", "Retained"],
            values=[churned_n, retained_n],
            hole=0.5,
            marker_colors=[COLOR_HIGH, COLOR_LOW],
            textinfo="label+percent",
            textfont=dict(size=14, family="Inter, Arial, sans-serif"),
        ))
        fig_donut.update_layout(
            template=TEMPLATE,
            font=PLOT_FONT,
            height=360,
            legend=dict(orientation="h", x=0.25, y=-0.05),
            margin=dict(l=20, r=20, t=20, b=40),
            annotations=[dict(
                text=f"<b>{churned_n:,}</b><br>Churned",
                x=0.5, y=0.5, font_size=16,
                font_family="Inter, Arial, sans-serif",
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_rev:
        st.subheader("💰 Revenue at Risk")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Monthly at Risk", f"${rev_monthly:,.0f}")
        with r2:
            st.metric("Annual at Risk", f"${rev_annual:,.0f}")
        with r3:
            st.metric("Avg Monthly (Churned)", f"${avg_m_churned:.2f}")

        st.markdown("""
| Metric | Value |
|--------|-------|
| Monthly revenue at risk | ${:,.0f} |
| Annual revenue at risk | ${:,.0f} |
| Churned customers | {:,} |
| Avg charge (churned) | ${:.2f}/mo |
| Potential savings (5pt reduction) | ${:,.0f}/yr |
""".format(rev_monthly, rev_annual, churned_n, avg_m_churned, rev_annual * 0.05))

    st.markdown("---")

    # ── Executive Summary ────────────────────────────────────────────────────
    with st.expander("📋 Executive Summary", expanded=False):
        st.markdown(f"""
### Key Findings

- **{churn_rate:.2f}% overall churn rate** — {churned_n:,} of {total_cust:,} customers churned.
- **${rev_monthly:,.0f}/month** (${rev_annual:,.0f}/year) in recurring revenue is at risk.
- **Month-to-month contracts** drive the highest churn at **42.71%** — 15× higher than two-year contracts.
- **Electronic check** users churn at **45.29%** vs ~16% for auto-pay customers.
- **Fiber optic** subscribers exhibit **41.89%** churn — quality/price perception gap likely driver.
- **New customers (0–12m)** churn at **47.44%** — the first year is the critical retention window.
- Churned customers have **higher average monthly charges** (${avg_m_churned:.2f} vs $61.27 for retained).
- Random Forest model achieves **{rf_accuracy*100:.1f}% accuracy** and **{rf_recall*100:.1f}% weighted recall**.
""")

    # ── Top 5 Recommendations ────────────────────────────────────────────────
    st.subheader("🎯 Top 5 Retention Recommendations")
    recs = [
        ("1️⃣ Contract Migration", "Offer month-to-month customers a 10% discount to upgrade to 12-month contracts. Targets the 42.71% churn segment — the single largest retention lever."),
        ("2️⃣ Auto-Pay Enrolment", "Provide a $5/month bill credit for switching from electronic check to automatic payment. The 45.29% churn rate drops to ~16% with auto-pay."),
        ("3️⃣ First-Year Onboarding", "Launch a 90-day structured onboarding programme for new customers. The 0–12m cohort has 47.44% churn — the highest of any tenure segment."),
        ("4️⃣ Fibre Quality Initiative", "Investigate and improve fibre optic service quality. At 41.89% churn, fibre subscribers represent significant revenue risk."),
        ("5️⃣ Predictive Retention Scoring", "Deploy the Random Forest model to score all active customers monthly. Prioritise proactive outreach for top-quartile risk customers."),
    ]
    for title, body in recs:
        st.info(f"**{title}** — {body}")

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — BUSINESS ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def page_business_analytics(df_clean, sql_data):
    st.markdown("# 📊 Business Analytics")
    st.markdown("**Churn rate breakdown by contract, payment method, internet service and tenure.**")

    q2 = sql_data["q2"]
    q3 = sql_data["q3"]
    q4 = sql_data["q4"]
    q5 = sql_data["q5"]

    # ── Row 1: Contract & Payment ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig_contract = make_hbar_chart(
            q2["Contract"].tolist(),
            q2["churn_rate_pct"].tolist(),
            "Churn Rate by Contract Type",
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    with col2:
        fig_payment = make_hbar_chart(
            q3["PaymentMethod"].tolist(),
            q3["churn_rate_pct"].tolist(),
            "Churn Rate by Payment Method",
        )
        st.plotly_chart(fig_payment, use_container_width=True)

    # ── Row 2: Internet & Tenure ──────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        fig_internet = make_hbar_chart(
            q4["InternetService"].tolist(),
            q4["churn_rate_pct"].tolist(),
            "Churn Rate by Internet Service",
        )
        st.plotly_chart(fig_internet, use_container_width=True)

    with col4:
        fig_tenure = make_hbar_chart(
            q5["tenure_bucket"].tolist(),
            q5["churn_rate_pct"].tolist(),
            "Churn Rate by Tenure Bucket",
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    # ── Heatmap: Contract × Tenure ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🗺️ Churn Rate Heatmap: Contract × Tenure Bucket")

    heatmap_data = (
        df_clean.groupby(["tenure_bucket", "Contract"])["churn_flag"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"churn_flag": "churn_rate_pct"})
    )
    heatmap_pivot = heatmap_data.pivot(
        index="tenure_bucket", columns="Contract", values="churn_rate_pct"
    ).reindex(["0-12m", "13-24m", "25-48m", "49-72m"])

    x_vals = list(heatmap_pivot.columns)
    y_vals = list(heatmap_pivot.index)
    z_vals = heatmap_pivot.values.tolist()
    text_vals = [[f"{v:.1f}%" if v is not None and not (isinstance(v, float) and v != v) else "" for v in row] for row in z_vals]

    fig_heatmap = go.Figure(go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        text=text_vals,
        texttemplate="%{text}",
        colorscale="Blues",
        colorbar=dict(title="Churn Rate (%)"),
    ))
    fig_heatmap.update_layout(
        template=TEMPLATE,
        font=PLOT_FONT,
        xaxis_title="Contract Type",
        yaxis_title="Tenure Bucket",
        height=380,
        margin=dict(l=80, r=40, t=30, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — CUSTOMER BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────────
def page_customer_behaviour(df_clean, sql_data):
    st.markdown("# 👥 Customer Behaviour")
    st.markdown("**Tenure and monthly charges distributions for churned vs retained customers.**")

    q5 = sql_data["q5"]

    churned_df  = df_clean[df_clean["Churn"] == "Yes"]
    retained_df = df_clean[df_clean["Churn"] == "No"]

    col1, col2 = st.columns(2)

    # ── Tenure Distribution ───────────────────────────────────────────────────
    with col1:
        st.subheader("Tenure Distribution")
        fig_tenure_dist = go.Figure()
        fig_tenure_dist.add_trace(go.Histogram(
            x=churned_df["tenure"],
            name="Churned",
            marker_color=COLOR_HIGH,
            opacity=0.75,
            nbinsx=30,
        ))
        fig_tenure_dist.add_trace(go.Histogram(
            x=retained_df["tenure"],
            name="Retained",
            marker_color=COLOR_LOW,
            opacity=0.75,
            nbinsx=30,
        ))
        fig_tenure_dist.update_layout(
            barmode="overlay",
            template=TEMPLATE,
            font=PLOT_FONT,
            xaxis_title="Tenure (months)",
            yaxis_title="Customer Count",
            legend=dict(orientation="h", x=0.3, y=1.02),
            height=360,
            margin=dict(l=20, r=20, t=60, b=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_tenure_dist, use_container_width=True)

    # ── Monthly Charges Distribution ──────────────────────────────────────────
    with col2:
        st.subheader("Monthly Charges Distribution")
        fig_charges_dist = go.Figure()
        fig_charges_dist.add_trace(go.Histogram(
            x=churned_df["MonthlyCharges"],
            name="Churned",
            marker_color=COLOR_HIGH,
            opacity=0.75,
            nbinsx=30,
        ))
        fig_charges_dist.add_trace(go.Histogram(
            x=retained_df["MonthlyCharges"],
            name="Retained",
            marker_color=COLOR_LOW,
            opacity=0.75,
            nbinsx=30,
        ))
        fig_charges_dist.update_layout(
            barmode="overlay",
            template=TEMPLATE,
            font=PLOT_FONT,
            xaxis_title="Monthly Charges ($)",
            yaxis_title="Customer Count",
            legend=dict(orientation="h", x=0.3, y=1.02),
            height=360,
            margin=dict(l=20, r=20, t=60, b=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_charges_dist, use_container_width=True)

    # ── Customer Count by Tenure Bucket ──────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Customer Counts by Tenure Bucket")
    q5_display = q5.copy()
    q5_display.columns = ["Tenure Bucket", "Customers", "Churned", "Churn Rate (%)"]
    st.dataframe(q5_display, use_container_width=True, hide_index=True)

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def page_machine_learning(ml_data):
    st.markdown("# 🤖 Machine Learning")
    st.markdown("**Logistic Regression vs Random Forest model comparison, feature importance and ROC curves.**")

    metrics_df  = ml_data["metrics_df"]
    rf_imp_df   = ml_data["rf_imp_df"]
    lr_coef_df  = ml_data["lr_coef_df"]
    cm_rf       = ml_data["cm_rf"]
    cm_lr       = ml_data["cm_lr"]
    fpr_lr      = ml_data["fpr_lr"]
    tpr_lr      = ml_data["tpr_lr"]
    auc_lr      = ml_data["auc_lr"]
    fpr_rf      = ml_data["fpr_rf"]
    tpr_rf      = ml_data["tpr_rf"]
    auc_rf      = ml_data["auc_rf"]

    # ── Model Comparison Table ────────────────────────────────────────────────
    st.subheader("📋 Model Comparison")

    def _highlight_rf(row):
        if row["Model"] == "Random Forest":
            return ["background-color: #E3F2FD; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        metrics_df.style.apply(_highlight_rf, axis=1).format({
            "Accuracy":       "{:.4f}",
            "Precision (wt)": "{:.4f}",
            "Recall (wt)":    "{:.4f}",
            "F1 (wt)":        "{:.4f}",
            "ROC-AUC":        "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.info("🏆 **Random Forest is recommended** — higher F1, Recall, and Accuracy. Crucial for identifying actual churners (higher recall minimises missed interventions).")

    st.markdown("---")

    # ── Feature Importance + Confusion Matrix ─────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RF Feature Importance (Top 10)")
        fig_imp = go.Figure(go.Bar(
            x=rf_imp_df["importance"].tolist(),
            y=rf_imp_df["feature"].tolist(),
            orientation="h",
            marker_color=COLOR_LOW,
            text=[f"{v:.3f}" for v in rf_imp_df["importance"].tolist()],
            textposition="outside",
        ))
        fig_imp.update_layout(
            template=TEMPLATE,
            font=PLOT_FONT,
            xaxis_title="Importance Score",
            yaxis=dict(autorange="reversed"),
            height=380,
            margin=dict(l=20, r=60, t=30, b=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.subheader("RF Confusion Matrix")
        cm_labels = ["Retained (0)", "Churned (1)"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_rf.tolist(),
            x=cm_labels,
            y=cm_labels,
            colorscale="Blues",
            text=[[str(v) for v in row] for row in cm_rf.tolist()],
            texttemplate="%{text}",
            textfont=dict(size=16, family="Inter, Arial, sans-serif"),
            showscale=True,
        ))
        fig_cm.update_layout(
            template=TEMPLATE,
            font=PLOT_FONT,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            yaxis=dict(autorange="reversed"),
            height=380,
            margin=dict(l=80, r=20, t=30, b=60),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # ── ROC Curves ────────────────────────────────────────────────────────────
    st.subheader("📈 ROC Curves — LR vs RF")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr_lr.tolist(), y=tpr_lr.tolist(),
        name=f"Logistic Regression (AUC={auc_lr:.4f})",
        line=dict(color="#FF7043", width=2),
        mode="lines",
    ))
    fig_roc.add_trace(go.Scatter(
        x=fpr_rf.tolist(), y=tpr_rf.tolist(),
        name=f"Random Forest (AUC={auc_rf:.4f})",
        line=dict(color=COLOR_LOW, width=2),
        mode="lines",
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random Classifier",
        line=dict(color="grey", width=1, dash="dash"),
        mode="lines",
        showlegend=True,
    ))
    fig_roc.update_layout(
        template=TEMPLATE,
        font=PLOT_FONT,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.1, bgcolor="rgba(255,255,255,0.8)", bordercolor="#E3E8F0", borderwidth=1),
        height=400,
        margin=dict(l=60, r=40, t=30, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ── LR Coefficients ───────────────────────────────────────────────────────
    st.subheader("LR Feature Coefficients (Top 15 by Magnitude)")
    lr_sorted = lr_coef_df.sort_values("abs_coef", ascending=True)
    coef_colors = [COLOR_HIGH if v < 0 else COLOR_LOW for v in lr_sorted["coefficient"].tolist()]

    fig_lr = go.Figure(go.Bar(
        x=lr_sorted["coefficient"].tolist(),
        y=lr_sorted["feature"].tolist(),
        orientation="h",
        marker_color=coef_colors,
        text=[f"{v:.3f}" for v in lr_sorted["coefficient"].tolist()],
        textposition="outside",
    ))
    fig_lr.update_layout(
        template=TEMPLATE,
        font=PLOT_FONT,
        xaxis_title="Coefficient Value",
        height=500,
        margin=dict(l=20, r=80, t=30, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(lr_sorted)-0.5, line=dict(color="black", width=1, dash="dot"))],
    )
    st.plotly_chart(fig_lr, use_container_width=True)
    st.caption("🔵 Blue = positive churn predictor (increases churn probability) | 🔴 Red = negative (reduces churn probability)")

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — AI BUSINESS ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────
def page_ai_assistant(df_clean, sql_data, ml_data):
    st.markdown("# 🧠 AI Business Assistant")
    st.markdown("**Powered by Google Gemini AI — with intelligent rule-based fallback.**")

    # ── Gemini API Key input ──────────────────────────────────────────────────
    gemini_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        key="gemini_key",
        placeholder="AIza...",
        help="Enter your Google Gemini API key. Leave blank to use rule-based fallback mode.",
    )

    gemini_model  = None
    gemini_avail  = False

    if gemini_key:
        try:
            import google.generativeai as genai  # noqa
            genai.configure(api_key=gemini_key)
            candidates = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
            for name in candidates:
                try:
                    _m = genai.GenerativeModel(name)
                    _m.generate_content("ping")
                    gemini_model = _m
                    gemini_avail = True
                    st.success(f"✅ Gemini configured — model: **{name}**")
                    break
                except Exception:
                    continue
            if not gemini_avail:
                st.warning("⚠️ Gemini API key provided but no model responded. Switching to Demo Mode.")
        except ImportError:
            st.warning("⚠️ `google-generativeai` package not installed. Using Demo Mode.")
        except Exception as e:
            st.warning(f"⚠️ Gemini unavailable: {e}. Using Demo Mode.")
    else:
        st.warning("⚠️ Running in **Demo Mode** — Gemini API key not configured. All responses use rule-based analytics engine.")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📋 Churn Drivers Summary",
        "💬 Ask a Business Question",
        "🔍 Explain Customer Risk",
    ])

    # ── Tab 1: Churn Drivers Summary ──────────────────────────────────────────
    with tab1:
        st.subheader("Churn Drivers Summary")
        st.markdown("Generate an executive summary of the top churn drivers based on the KPI analysis.")

        if st.button("🚀 Generate Executive Summary", key="btn_summary"):
            kpi_tables = {
                "q1_result": sql_data["q1"],
                "q2_result": sql_data["q2"],
                "q3_result": sql_data["q3"],
                "q4_result": sql_data["q4"],
                "q5_result": sql_data["q5"],
                "q6_result": sql_data["q6"],
            }
            with st.spinner("Generating summary..."):
                summary = summarize_churn_drivers(kpi_tables, gemini_model=gemini_model)

            provider = summary.get("provider", "rule-based")
            st.caption(f"Provider: **{provider}**")

            exec_summary = summary.get("executive_summary", "")
            if isinstance(exec_summary, str):
                st.markdown(exec_summary)
            else:
                st.markdown(str(exec_summary))

            drivers = summary.get("churn_drivers", [])
            if isinstance(drivers, list) and drivers:
                st.markdown("### 🔍 Top Churn Drivers")
                for i, d in enumerate(drivers, 1):
                    st.markdown(f"""
**{i}. {d.get('driver', '')}**
- Churn rate: `{d.get('churn_rate', '')}`  |  Benchmark: `{d.get('benchmark', '')}`
- Impact: {d.get('impact', '')}
""")

            recs = summary.get("recommendations", [])
            if isinstance(recs, list) and recs:
                st.markdown("### 💡 Recommendations")
                for r in recs:
                    st.markdown(f"- {r}")

    # ── Tab 2: Ask a Business Question ───────────────────────────────────────
    with tab2:
        st.subheader("Ask a Business Question")
        st.markdown("Type a natural-language business question. The assistant will generate SQL and execute it.")

        question = st.text_input(
            "Your question",
            value="Which contract type has the highest churn?",
            key="biz_question",
        )

        if st.button("🔍 Generate Answer", key="btn_question"):
            if question.strip():
                with st.spinner("Generating SQL and fetching results..."):
                    answer = ask_churn_question(question, df_clean, gemini_model=gemini_model)

                st.markdown(f"**❓ Question:** {answer.get('question', '')}")

                sql_str = answer.get("sql", "")
                if sql_str:
                    st.markdown("**🔎 SQL Query:**")
                    st.code(sql_str, language="sql")

                results_df = answer.get("results", pd.DataFrame())
                if results_df is not None and not results_df.empty:
                    st.markdown("**📊 Results:**")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                explanation = answer.get("explanation", "")
                if explanation:
                    st.markdown(f"**💬 Explanation:** {explanation}")
            else:
                st.warning("Please enter a question first.")

    # ── Tab 3: Explain Customer Risk ──────────────────────────────────────────
    with tab3:
        st.subheader("Explain Customer Risk")
        st.markdown("Enter a customer ID to get their churn probability and personalised retention recommendation.")

        customer_id = st.text_input(
            "Customer ID",
            value="9237-HQITU",
            key="cust_id",
            placeholder="e.g. 9237-HQITU",
        )

        if st.button("🔍 Analyze Customer", key="btn_cust"):
            if customer_id.strip():
                with st.spinner("Scoring customer..."):
                    risk_info = explain_customer_risk(
                        customer_id,
                        df_clean,
                        ml_data["rf_pipeline"],
                        FEATURE_COLS,
                    )

                st.markdown(f"### Customer Risk Report — `{risk_info.get('customer_id', '')}`")

                prob = risk_info.get("churn_probability")
                risk_level = risk_info.get("risk_level", "Unknown")

                col_prob, col_risk = st.columns(2)
                with col_prob:
                    if prob is not None:
                        st.metric("Churn Probability", f"{prob:.1f}%")
                    else:
                        st.metric("Churn Probability", "N/A")
                with col_risk:
                    st.metric("Risk Level", risk_level)

                st.markdown(f"**Profile:** {risk_info.get('profile_summary', 'N/A')}")

                factors = risk_info.get("key_risk_factors", [])
                if factors:
                    st.markdown("**🔑 Key Risk Factors:**")
                    for f in factors:
                        st.markdown(f"  - {f}")

                st.markdown(f"**💡 Retention Recommendation:** {risk_info.get('retention_recommendation', 'N/A')}")
                st.markdown(f"**🎁 Marketing Offer:** {risk_info.get('marketing_offer', 'N/A')}")
            else:
                st.warning("Please enter a customer ID.")

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📊 Telco Churn Dashboard")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            options=[
                "🏠 Executive Overview",
                "📊 Business Analytics",
                "👥 Customer Behaviour",
                "🤖 Machine Learning",
                "🧠 AI Business Assistant",
            ],
            index=0,
        )
        st.markdown("---")
        st.caption("Data: WA Telco Customer Churn")
        st.caption("Records: 7,043 customers")
        st.caption("Model: Random Forest (AUC=0.830)")

    # ── Load data & models ────────────────────────────────────────────────────
    try:
        with st.spinner("Loading data..."):
            df_clean = load_and_clean_data()
        
        with st.spinner("Running SQL analytics..."):
            sql_data = run_sql_queries(df_clean)
        
        with st.spinner("Training ML models..."):
            ml_data = train_models(df_clean)
        
        # ── Route to page ───────────────────────────────
        if page == "🏠 Executive Overview":
            page_executive_overview(df_clean, sql_data, ml_data)
        elif page == "📊 Business Analytics":
            page_business_analytics(df_clean, sql_data)
        elif page == "👥 Customer Behaviour":
            page_customer_behaviour(df_clean, sql_data)
        elif page == "🤖 Machine Learning":
            page_machine_learning(ml_data)
        elif page == "🧠 AI Business Assistant":
            page_ai_assistant(df_clean, sql_data, ml_data)

    except Exception as e:
        st.exception(e)
        st.stop()


if __name__ == "__main__":
    main()