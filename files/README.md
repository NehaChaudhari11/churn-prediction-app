# 📡 ChurnRadar — Telecom Customer Churn Predictor

Built by **Neha Chaudhari** | AI & Data Science | B.E. Final Year

---

## 🚀 What This Does

Upload a CSV of telecom customer data and instantly get:
- ✅ Churn prediction for every customer (Will Churn / Will Stay)
- 📊 Churn probability score (0–100%)
- 🔴 Risk level classification (HIGH / MEDIUM / LOW)
- 📈 Visual breakdown by contract type, risk distribution, and probability buckets
- ⬇ Downloadable results CSV

**Model:** Random Forest Classifier | **Accuracy:** 91.5%

---

## 🛠 How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open browser
Go to `http://localhost:8501`

---

## ☁ Deploy to Streamlit Cloud (Free)

1. Push this folder to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → connect your repo → set `app.py` as the main file
4. Click **Deploy** — your app will be live in ~2 minutes!

---

## 📁 Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `churn_model.pkl` | Trained Random Forest model |
| `churn_encoders.pkl` | Label encoders for categorical features |
| `feature_cols.pkl` | Feature column list |
| `requirements.txt` | Python dependencies |
| `sample_telecom_customers.csv` | Sample input file |

---

## 📋 Required CSV Columns

| Column | Type | Example |
|--------|------|---------|
| tenure | number | 24 |
| MonthlyCharges | number | 79.85 |
| TotalCharges | number | 1916.4 |
| SeniorCitizen | 0 or 1 | 0 |
| gender | Male / Female | Female |
| Partner | Yes / No | Yes |
| Dependents | Yes / No | No |
| PhoneService | Yes / No | Yes |
| MultipleLines | Yes / No / No phone service | No |
| InternetService | DSL / Fiber optic / No | Fiber optic |
| OnlineSecurity | Yes / No / No internet service | No |
| TechSupport | Yes / No / No internet service | No |
| Contract | Month-to-month / One year / Two year | Month-to-month |
| PaperlessBilling | Yes / No | Yes |
| PaymentMethod | Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic) | Electronic check |

---

## 🧠 Tech Stack

- **Python** — Core language
- **Scikit-learn** — Machine Learning (Random Forest)
- **Pandas / NumPy** — Data processing
- **Streamlit** — Web app framework

---

*This project was built as part of a portfolio to demonstrate end-to-end ML deployment skills.*
