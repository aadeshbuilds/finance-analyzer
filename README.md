<div align="center">

# 🛡️ FinShield AI
### AI-Powered Personal Finance Analyzer & Smart Anomaly Detection System

[![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**[🚀 Live Demo](https://finshield-ai.streamlit.app)** • **[📊 Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)** • **[📧 Contact](mailto:aadeshsenpai@gmail.com)**

</div>

---

## 📌 Overview

FinShield AI is a **production-grade machine learning system** for detecting fraudulent financial transactions and analyzing spending patterns. Built entirely from scratch using Python, it demonstrates the complete ML engineering workflow — from raw data to deployed web application.

> ⚠️ **Disclaimer:** This is a demonstration project built for educational purposes. Results should not be used as the sole basis for fraud detection decisions.

---

## 🎯 Key Results

| Metric | Score |
|--------|-------|
| 🎯 ROC-AUC | **0.9998** |
| ⚡ F1 Score | **0.9991** |
| 🎪 Precision | **1.0000** |
| 🔍 Recall | **0.9982** |
| 📊 Training Size | **58,213 transactions** |

---

## ✨ Features

- **🛡️ Fraud Detection** — XGBoost + Random Forest ensemble with 99.98% AUC
- **📊 Visual Analytics** — Interactive Plotly dashboards with real-time charts
- **🔍 Single Transaction Analyzer** — Instant fraud scoring with gauge visualization
- **📁 Batch Analysis** — Upload CSV and analyze thousands of transactions at once
- **⬇️ Downloadable Reports** — Export full analysis as CSV
- **🌙 Dark Professional UI** — Modern fintech-style dashboard

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.14 |
| ML Models | XGBoost, Random Forest, Logistic Regression |
| ML Framework | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Serialization | Joblib |
| Deployment | Streamlit Cloud |

---

## 🤖 ML Pipeline

Raw Data (6.3M transactions)
↓
Stratified Sampling (58K rows)
↓
Data Cleaning & Preprocessing
↓
Feature Engineering (27 features)
↓
Model Training + Hyperparameter Tuning
↓
Evaluation (ROC-AUC, F1, Confusion Matrix)
↓
Streamlit Dashboard
↓
Streamlit Cloud Deployment

---

## 📊 Models Trained

| Model | ROC-AUC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| **XGBoost** ⭐ | 0.9998 | 0.9991 | 1.0000 | 0.9982 |
| Random Forest | 0.9998 | 0.9991 | 1.0000 | 0.9982 |
| Logistic Regression | 0.9982 | 0.9461 | 0.9167 | 0.9775 |

---

## 🧠 Feature Engineering

27 features engineered across 5 categories:

| Category | Features |
|----------|---------|
| Time-based | `hour_of_day`, `day_of_simulation`, `is_late_night` |
| Transaction Risk | `is_high_risk_type`, `is_transfer`, `is_cash_out` |
| Balance Mismatch | `balance_mismatch_orig`, `has_balance_mismatch` |
| Amount Analysis | `amount_zscore`, `is_large_transaction`, `log_amount` |
| Receiver Risk | `receiver_was_empty`, `large_amount_to_empty` |

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/aadeshbuilds/finance-analyzer.git
cd finance-analyzer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

finance-analyzer/
├── app/
│   └── streamlit_app.py      # Main Streamlit dashboard
├── src/
│   ├── preprocessing.py      # Data cleaning pipeline
│   ├── features.py           # Feature engineering
│   ├── train.py              # Model training + tuning
│   ├── evaluate.py           # Model evaluation + charts
│   └── model_utils.py        # Prediction utilities
├── models/
│   ├── best_model_pipeline.pkl
│   ├── xgboost_pipeline.pkl
│   └── model_config.json
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned data
├── notebooks/
│   └── eda.ipynb             # Exploratory analysis
├── reports/                  # Generated charts
├── requirements.txt
└── README.md

---

## 🗺️ Roadmap

- [x] Data preprocessing pipeline
- [x] Feature engineering (27 features)
- [x] Multi-model training with hyperparameter tuning
- [x] Professional evaluation metrics
- [x] Streamlit dashboard
- [x] Cloud deployment
- [ ] Column mapper for any bank CSV format
- [ ] PDF report generation
- [ ] Real-time transaction monitoring
- [ ] Email alerts for high-risk transactions

---

## 👨‍💻 Author

**Aadesh Ambadas Kadam**

[![GitHub](https://img.shields.io/badge/GitHub-aadeshbuilds-black?style=flat&logo=github)](https://github.com/aadeshbuilds)

---

<div align="center">
Built with ❤️ using Python & Streamlit
</div>

