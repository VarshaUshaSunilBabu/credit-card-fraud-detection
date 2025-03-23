# 💳 Credit Card Fraud Detection Using XGBoost

Built an end-to-end machine learning solution to detect fraudulent credit card transactions on an imbalanced dataset, using SMOTE for class balancing and SHAP for explainable AI.

---

## 📌 Overview

- **Goal:** Detect fraud with high accuracy while minimizing false positives.
- **Model:** XGBoost Classifier with threshold tuning.
- **Explainability:** SHAP for global and local interpretability.
- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📊 Dataset Summary

- **Transactions:** 284,807
- **Fraud Cases:** 492 (~0.17%)
- **Features:** 30 PCA-transformed numerical features + Amount
- **Target:** `Class` (0 = Legit, 1 = Fraud)

---

## ⚙️ Technologies Used

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- SMOTE (imbalanced-learn)
- SHAP for interpretability
- Plotly, Matplotlib for visualizations

---

## 🧠 Key Results

| Metric      | Before Threshold Tuning | After Threshold Tuning |
|-------------|--------------------------|-------------------------|
| Precision   | 17.84%                   | 82.01% 🔥               |
| Recall      | 85.81%                   | 77.03% ✅               |
| F1-Score    | 29.53%                   | 79.44% 🚀               |
| AUC Score   | 0.9694                   | 0.9694                  |

Threshold tuning significantly improved **precision** without sacrificing much **recall** — ideal for fraud detection where false positives are costly.

---

## 📈 Visualizations

- 📌 SHAP Summary Plot  
- 🔍 SHAP Waterfall (single prediction explanation)  
- 📊 XGBoost Feature Importance  
- 💰 Fraud Transaction Amount Distribution  
- ⚖️ Risk Score Distribution


## 💾 Model Deployment

Model saved as `xgboost_fraud_model.pkl` for reuse or deployment in web apps or APIs.

```python
import pickle
with open("xgboost_fraud_model.pkl", "rb") as file:
    model = pickle.load(file)
