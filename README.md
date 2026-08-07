# 📊 Telco Customer Churn Dashboard

An end-to-end Business Intelligence and Machine Learning solution for customer churn analysis, prediction, and decision support. This project combines data engineering, SQL analytics, predictive machine learning, interactive visualization, and Generative AI to help business stakeholders identify churn drivers, estimate revenue at risk, and develop targeted customer retention strategies.

📑 Table of Contents
- Project Overview
- Business Objectives
- Business Questions Answered
- Dataset
- Technologies Used
- Project Workflow
- Dashboard Features
- Key Business Insights
- Business Recommendations
- Dashboard Preview
- Installation
- Project Structure
- Future Improvements
- License
- Acknowledgements
- Contact
---
🚀 Project Overview

Customer churn is one of the most significant challenges faced by subscription-based businesses. Losing existing customers directly impacts recurring revenue and profitability.

This project demonstrates a complete analytics workflow using the IBM Telco Customer Churn dataset, transforming raw customer data into actionable business insights through:

- Data cleaning and preprocessing
- SQL-based business analytics
- Exploratory Data Analysis (EDA)
- Machine Learning prediction
- Interactive Business Intelligence dashboards
- AI-powered business assistant
- Executive reporting and recommendations

---

## 🎯 Business Objectives

The project aims to:

- Identify the primary drivers of customer churn
- Predict customers likely to churn using machine learning
- Quantify revenue at risk due to churn
- Support data-driven customer retention strategies
- Deliver insights through an interactive dashboard

---
 ## ❓ Business Questions Answered

This project addresses the following key business questions:

1. **What is the overall customer churn rate?**
   - How many customers have churned versus those retained?

2. **Which customer segments are most likely to churn?**
   - By contract type
   - By payment method
   - By internet service
   - By tenure
   - By demographic characteristics

3. **Which factors contribute most to customer churn?**
   - What customer attributes have the greatest impact on churn probability?

4. **How much revenue is at risk due to customer churn?**
   - Monthly revenue at risk
   - Annual revenue at risk
   - Financial impact of customer attrition

5. **Can customer churn be predicted accurately?**
   - Compare Logistic Regression and Random Forest models.
   - Evaluate predictive performance using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrices.

6. **Which machine learning model performs best?**
   - Identify the most effective model for churn prediction.

7. **Which customers are at the highest risk of leaving?**
   - Explain individual customer risk using model predictions and AI-generated insights.

8. **What business actions can reduce customer churn?**
   - Identify actionable retention strategies based on analytical findings.

9. **How can AI support business decision-making?**
   - Generate executive summaries.
   - Answer natural language business questions.
   - Explain customer churn risk.
   - Recommend data-driven retention strategies.

---

## 📂 Dataset

**Source**

IBM Telco Customer Churn Dataset

**Dataset Summary**

- 7,043 customer records
- 21 customer attributes
- Binary target variable (`Churn`)

### Key Variables

- Customer demographics
- Contract type
- Payment method
- Internet service
- Monthly charges
- Total charges
- Tenure
- Online services
- Streaming services
- Churn status

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| Programming | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| SQL Analytics | DuckDB |
| Machine Learning | Scikit-learn |
| Visualisation | Plotly |
| AI | Google Gemini API |
| Version Control | Git & GitHub |

---

## 📈 Project Workflow

### 1. Data Preparation

- Data loading
- Missing value handling
- Data cleaning
- Data type conversion
- Feature engineering

---

### 2. Exploratory Data Analysis

Performed analysis on:

- Customer demographics
- Contract types
- Payment methods
- Internet services
- Monthly charges
- Customer tenure
- Churn distribution

---

### 3. SQL Business Analytics

Business insights were generated using DuckDB, including:

- Overall churn rate
- Churn by contract type
- Churn by payment method
- Churn by internet service
- Customer tenure analysis
- Revenue at risk
- Customer segmentation

---

### 4. Machine Learning

Two classification models were developed:

- Logistic Regression
- Random Forest Classifier

Model evaluation included:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score
- Confusion Matrix
- Feature Importance

---

### 5. AI Business Intelligence

Integrated a Generative AI assistant capable of:

- Executive business summaries
- Natural language business questions
- Customer risk explanations
- Business recommendations

Supports:

- Google Gemini API
- Automatic Demo Mode fallback

---

## 📊 Dashboard Features

### 🏠 Executive Overview

- KPI Cards
- Churn Rate
- Revenue at Risk
- Customer Summary
- Executive Recommendations

---

### 📊 Business Analytics

- Contract Analysis
- Payment Method Analysis
- Internet Service Analysis
- Revenue Analysis
- SQL Insights

---

### 👥 Customer Behaviour

Interactive visualisations including:

- Monthly Charges
- Tenure
- Senior Citizens
- Dependents
- Service Usage
- Customer Segments

---

### 🤖 Machine Learning

- Model Comparison
- ROC Curves
- Confusion Matrices
- Feature Importance
- Logistic Regression Coefficients

---

### 🧠 AI Business Assistant

Supports:

- Executive summaries
- Business Q&A
- Customer risk explanation
- AI-generated retention recommendations

---

## 📌 Key Business Insights

The analysis identified several significant churn drivers:

- Month-to-month contracts exhibited the highest churn rates.
- Electronic check payment methods were strongly associated with customer churn.
- Fibre optic customers experienced higher churn than other service types.
- Customers within their first year represented the highest-risk group.
- Significant recurring revenue was identified as being at risk due to customer churn.

---

## 💡 Business Recommendations

- Encourage customers to migrate from month-to-month to longer-term contracts.
- Promote automatic payment methods.
- Improve first-year customer onboarding.
- Investigate Fibre Optic service quality.
- Deploy predictive churn scoring for proactive retention campaigns.

---

## 📷 Dashboard Preview

## 📷 Dashboard Preview

### 🏠 Executive Overview

![Executive Overview](screenshots/executive-overview.png)

---

### 📊 Business Analytics

![Business Analytics](screenshots/business-analytics.png)

---

### 👥 Customer Behaviour

![Customer Behaviour](screenshots/customer-behaviour.png)

---

### 🤖 Machine Learning

![Machine Learning](screenshots/machine-learning.png)

---

### 🧠 AI Business Assistant

![AI Business Assistant](screenshots/ai-assistant.png)

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/telco-customer-churn-dashboard.git
```

Navigate to the project

```bash
cd telco-customer-churn-dashboard
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Telco-Customer-Churn-Dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── .streamlit/
│   └── config.toml
└── screenshots/
```

---

## 🔮 Future Improvements

Potential enhancements include:

- Real-time customer scoring
- CRM integration
- Automated retention campaigns
- XGBoost and LightGBM model comparison
- Explainable AI (SHAP)
- Cloud deployment
- Scheduled model retraining

---

## 📜 License

This project is provided for educational and portfolio purposes.

---

## 👨‍💻 Author

**Your Name**

GitHub: https://github.com/gbeminiyiprecious721-cmd

LinkedIn: https://www.linkedin.com/in/gbeminiyi-awolade-21257819a

---

## ⭐ Acknowledgements

- IBM Telco Customer Churn Dataset
- Streamlit
- Scikit-learn
- DuckDB
- Plotly
- Google Gemini API

---

## 📬 Contact

If you have questions, suggestions, or feedback, feel free to connect through GitHub or LinkedIn.

---

**If you found this project useful, please consider giving it a ⭐ on GitHub.**
