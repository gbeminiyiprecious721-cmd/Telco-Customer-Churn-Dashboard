# Telco Customer Churn Dashboard

### AI-Powered Business Intelligence with Python, SQL & Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?logo=plotly&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-0.10%2B-FFF000?logo=duckdb&logoColor=black)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-4285F4?logo=google&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

1. [Overview](#overview)
2. [Live Demo](#live-demo)
3. [Project Structure](#project-structure)
4. [Key Findings](#key-findings)
5. [Project Report](#project-report)
6. [Dashboard Pages](#dashboard-pages)
7. [Tech Stack](#tech-stack)
8. [Installation & Local Setup](#installation--local-setup)
9. [Dataset](#dataset)
10. [AI Assistant Setup](#ai-assistant-setup)
11. [Deployment (Streamlit Community Cloud)](#deployment-streamlit-community-cloud)
12. [.gitignore Recommendations](#gitignore-recommendations)
13. [License](#license)
14. [Author](#author)

---

## Overview

The **Telco Customer Churn Dashboard** is a full-stack, production-ready data intelligence application that combines SQL analytics, machine learning, and generative AI to help telecommunications companies understand, predict, and act on customer churn.

**Who it's for:**
- **Data Scientists** — Explore two end-to-end ML pipelines (Logistic Regression + Random Forest), feature engineering decisions, and model comparison methodology.
- **Business Analysts** — Slice churn by contract type, payment method, internet service, and tenure without writing a single line of code.
- **Hiring Managers** — A concise demonstration of the full data-science lifecycle: data ingestion → SQL analysis → machine learning → AI-powered insights → deployed Streamlit application.

**Key capabilities:**
| Capability | Details |
|---|---|
| 🗄️ **SQL Analytics** | 9 pre-built DuckDB queries covering churn rate, revenue at risk, and top-risk segments |
| 🤖 **ML Churn Prediction** | Logistic Regression and Random Forest classifiers with ROC-AUC > 0.83 |
| 💬 **AI BI Assistant** | Google Gemini-powered natural-language interface; rule-based Demo Mode requires no API key |
| 📊 **Interactive Dashboard** | 5-page Streamlit app with Plotly charts, KPI cards, confusion matrices, and feature importance |

---

## Live Demo

> 🚀 **[Launch App → https://your-app.streamlit.app](https://your-app.streamlit.app)**

To deploy your own instance, see [Deployment (Streamlit Community Cloud)](#deployment-streamlit-community-cloud).

<!-- 📸 Screenshot placeholder — add a GIF or PNG of the dashboard here
![Dashboard Screenshot](docs/screenshot.png)
-->

---

## Project Structure

```
Telco-Customer-Churn-Dashboard/
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (not included)
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
└── notebooks/
    └── telco_churn_analysis.ipynb  # Full analysis notebook (optional)
```

---

## Key Findings

> All metrics are derived from the IBM Telco Customer Churn dataset (7,043 customers).

| Metric | Value |
|---|---|
| **Overall churn rate** | **26.54%** (1,869 of 7,043 customers) |
| **Revenue at risk** | **$139,130 / month → $1,669,570 / year** |
| **Month-to-month vs Two-year churn** | 42.71% vs 2.83% — **15× higher risk** |
| **Highest-churn payment method** | Electronic check — **45.29%** |
| **Highest-churn internet service** | Fiber optic — **41.89%** |
| **Highest-churn tenure bucket** | 0–12 months — **47.44%** |

## 📄 Project Report

A detailed report documenting the project methodology, analysis, machine learning models, results, and business recommendations is available here:

🔗 **Project Report:** [https://your-report-link](https://app.zerve.ai/report/4cf27dfd-b005-4965-ad4e-03d7f305fcba)

### Machine Learning Model Comparison

| Model | Accuracy | F1 Score (weighted) | ROC-AUC |
|---|---|---|---|
| Random Forest | **77.1%** | **0.7755** | 0.8302 |
| Logistic Regression | 73.5% | 0.7490 | **0.8383** |

*Random Forest recommended for production: higher F1 and recall, reducing missed churners.*

---

## Dashboard Pages

### 1. 📊 Executive Overview
High-level command centre for leadership and stakeholders.
- **KPI cards**: Overall churn rate (26.54%), monthly revenue at risk ($139K), best model accuracy (77.1%)
- **Churn distribution** donut chart (retained vs churned)
- **Revenue at risk** breakdown table by customer segment
- **Executive summary** narrative with the top 3 churn drivers
- **Top 5 retention recommendations** ranked by estimated impact (contract upgrades, auto-pay migration, fibre service improvements)

### 2. 📈 Business Analytics
Deep-dive into what drives churn — filterable, interactive charts.
- Churn rate by **Contract Type** (bar chart highlighting the 15× month-to-month premium)
- Churn rate by **Payment Method** (electronic check vs auto-payment gap)
- Churn rate by **Internet Service** (Fiber optic vs DSL vs None)
- Churn rate by **Tenure Bucket** (new customers at highest risk)
- **Tenure × Contract interaction heatmap** for cross-segment analysis

### 3. 👤 Customer Behaviour
Distributional views of the customer population.
- **Tenure distribution** histograms: churned vs retained customers overlaid
- **Monthly charges distribution**: churned customers skew toward higher-priced plans
- Annotations highlight median values and at-risk thresholds

### 4. 🤖 Machine Learning
Full model transparency for technical stakeholders.
- **Model comparison table**: Accuracy, Precision, Recall, F1, ROC-AUC side-by-side
- **Confusion matrices** for Logistic Regression and Random Forest
- **ROC curves** with AUC annotations
- **Feature importance** (Random Forest) and **coefficient chart** (Logistic Regression)
- Business interpretation of each top predictor

### 5. 💬 AI Business Assistant
Conversational business intelligence powered by Google Gemini (with rule-based fallback — no API key required for Demo Mode).
- **Summarise churn drivers**: one-click executive summary of all KPI tables
- **Natural language → SQL**: ask any business question in plain English; the assistant generates and executes a safe DuckDB query
- **Explain customer risk**: enter a Customer ID to get an individual risk score, probability estimate, and tailored retention offer (discount, contract upgrade, service review)

---

## Tech Stack

| Technology | Purpose | Version |
|---|---|---|
| **Python** | Core language | 3.10+ |
| **Streamlit** | Web application framework | 1.x |
| **Plotly** | Interactive visualisations | 5.x |
| **Pandas** | Data manipulation and analysis | 2.x |
| **NumPy** | Numerical computing | 1.x |
| **DuckDB** | In-process SQL analytics engine | 0.10+ |
| **Scikit-learn** | Machine learning (LR + RF pipelines) | 1.x |
| **Google Gemini** | AI-powered insights and NL→SQL (optional) | `google-generativeai` |

---

## Installation & Local Setup

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Telco-Customer-Churn-Dashboard.git
cd Telco-Customer-Churn-Dashboard

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset
#    Download from Kaggle (see Dataset section) and place here:
#    data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 5. Launch the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | IBM Telco Customer Churn |
| **Platform** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **Rows** | 7,043 customers |
| **Features** | 21 columns (demographics, services, billing) |
| **Target** | `Churn` — binary Yes/No |

> ⚠️ **Do NOT commit the CSV to the repository.** Add `data/` to your `.gitignore` (see [.gitignore Recommendations](#gitignore-recommendations)).

---

## AI Assistant Setup

The AI Business Assistant works in **Demo Mode** by default using rule-based logic — no API key is needed.

To unlock the full Gemini-powered experience:

### 1. Get a Google API Key
Visit [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and create a free API key.

### 2. Local development
```bash
export GOOGLE_API_KEY=your-key-here
streamlit run app.py
```

Or add it to a `.env` file (never commit this file):
```
GOOGLE_API_KEY=your-key-here
```

### 3. Streamlit Community Cloud
1. Open your app's settings in [share.streamlit.io](https://share.streamlit.io)
2. Navigate to **Secrets**
3. Add:
```toml
GOOGLE_API_KEY = "your-key-here"
```

> The app detects the key at startup and switches automatically from Demo Mode to Gemini mode.

---

## Deployment (Streamlit Community Cloud)

Streamlit Community Cloud offers **free hosting** for public GitHub repositories.

1. **Fork or push** this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository and branch.
4. Set **Main file path** to `app.py`.
5. (Optional) Under **Advanced settings → Secrets**, add your `GOOGLE_API_KEY`.
6. Click **Deploy** — your app will be live within a few minutes.

> 📌 Ensure `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` is present in the repo **or** use Streamlit's `st.file_uploader` to load it at runtime (not included in this version).

---

## .gitignore Recommendations

Add the following to your `.gitignore` to avoid committing sensitive or large files:

```gitignore
# Dataset (too large and not redistributable)
data/
*.csv

# Environment variables / secrets
.env
.streamlit/secrets.toml

# Python bytecode
__pycache__/
*.pyc
*.pyo

# Virtual environments
.venv/
venv/
env/

# OS artefacts
.DS_Store
Thumbs.db
```

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Author

Created by Gbeminiyi Precious A.

---

*Built with ❤️ using Python, Streamlit, and the IBM Telco Customer Churn dataset.*
