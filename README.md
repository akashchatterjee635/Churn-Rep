This project aims to predict customer churn in a telecom dataset using advanced machine learning techniques. It includes exploratory data analysis, robust feature engineering, model tuning, threshold optimization, and explainability using SHAP.

---

## 🔍 Problem Statement

Customer churn is a critical issue for subscription-based businesses. The goal is to build a predictive model that accurately classifies whether a customer is likely to churn, enabling proactive retention strategies.

---

## 📦 Dataset

- **Source**: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size**: ~7,000 rows × 20+ columns
- **Target variable**: `Churn` (Yes/No)

---

## ⚙️ Project Pipeline

### ✅ 1. Exploratory Data Analysis (EDA)
- Handled missing values and data types
- Visualized churn distribution, service usage, contract types
- Detected class imbalance (74% No churn, 26% Yes)

### 🛠️ 2. Feature Engineering
- Cleaned and converted `TotalCharges`
- One-hot encoded categorical variables
- Created:
  - `TotalServices`: total services subscribed
  - `AvgMonthly`: TotalCharges / tenure
  - `AutoPay` flag
  - `LongTermCustomer` indicator

### 📊 3. Model Training (XGBoost)
- Applied `scale_pos_weight` to handle imbalance
- Used `XGBClassifier` with:
  - `eval_metric='logloss'`
  - `use_label_encoder=False`
- Trained on train-test split (80/20)

### 🎯 4. Threshold Optimization
- Evaluated multiple thresholds on validation set
- **Best result at threshold = 0.40**:
  - **Accuracy**: 74.3%
  - **F1-score**: 0.591
  - **ROC-AUC**: 0.802

### 🧠 5. Explainability with SHAP
- Used `shap.Explainer` on test set
- Visualized global feature importance (summary plot)
- Interpreted individual churn predictions via force plots

---

## 📈 Key Results

| Metric      | Value   |
|-------------|---------|
| Accuracy    | 0.743   |
| F1-Score    | 0.591   |
| ROC-AUC     | 0.802   |

Top Predictive Features (via SHAP):
- `Contract_TwoYear`
- `tenure`
- `AutoPay`
- `MonthlyCharges`

---

## 🧪 Tech Stack

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- XGBoost
- SHAP
- Jupyter Notebook

---

## 📌 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/telecom-churn-xgboost.git
cd telecom-churn-xgboost

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Churn_Prediction_XGBoost.ipynb
