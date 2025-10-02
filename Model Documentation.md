

# 📄 Model Documentation

## 1. Dataset & Features

We worked with a **custom dataset of loan beneficiaries**. Each record corresponds to one beneficiary, identified by `beneficiary_id`, with the following key attributes:

* **loan_amount** → Total loan applied for.
* **tenure** → Loan repayment period (in months).
* **repayments_on_time_ratio** → Fraction of repayments made on time.
* **num_past_loans** → Number of past loans taken by the customer.
* **outstanding_amount** → Pending loan amount (if any).
* **avg_monthly_kwh** → Average monthly electricity consumption.
* **mobile_recharge_avg** → Average monthly recharge amount.
* **utility_bill_on_time_ratio** → Ratio of timely bill payments.
* **business_sector** → Sector in which the beneficiary works (categorical).
* **default** → Whether the customer defaulted earlier (0 = No, 1 = Yes).
* **CBSC (Credit Behavior Score for Customers)** → Target variable (continuous score between ~0–100).

The **CBSC** score is the label our model predicts.


## 2. Preprocessing

Before training, we applied several preprocessing steps:

1. **Handling categorical features**:

   * `business_sector` was one-hot encoded using `pd.get_dummies`.

2. **Scaling numeric features**:

   * Continuous variables were standardized using `StandardScaler` to improve model performance.

3. **Feature/Target Split**:

   ```python
   X = df.drop(['CBSC', 'beneficiary_id'], axis=1)
   y = df['CBSC'].values
   ```

4. **Train-Test Split**:

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

---

## 3. Model Training (XGBoost)

We trained an **XGBoost Regressor** because CBSC is a **continuous score**, not a classification label.

```python
from xgboost import XGBRegressor

model_xgb = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

model_xgb.fit(X_train, y_train)
```

We saved the trained model using `joblib` for deployment:

```python
import joblib
joblib.dump(model_xgb, "xgb_cbsc_model.pkl")
```

---

## 4. Prediction Pipeline

When new **beneficiary data** comes in from the frontend:

1. **Preprocessing**:

   * Convert to DataFrame
   * One-hot encode `business_sector`
   * Reindex columns to match training features
   * Apply the same `StandardScaler`

   ```python
   test_record = pd.DataFrame([new_data])
   test_record = pd.get_dummies(test_record, drop_first=True)
   test_record = test_record.reindex(columns=X_train.columns, fill_value=0)
   test_scaled = sc.transform(test_record)
   ```

2. **Prediction**:

   ```python
   predicted_score = model_xgb.predict(test_scaled)[0]
   ```

3. **Risk Categorization**:

   * **Low Risk** → CBSC ≥ 70
   * **Medium Risk** → 50 ≤ CBSC < 70
   * **High Risk** → CBSC < 50

   ```python
   if predicted_score >= 70:
       risk = "Low Risk"
   elif predicted_score >= 50:
       risk = "Medium Risk"
   else:
       risk = "High Risk"
   ```

---

## 5. Explainability with SHAP

To make the model interpretable, we used **SHAP (SHapley Additive exPlanations)**:

```python
import shap
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(test_scaled)
```

* SHAP gives **feature importance per prediction**.
* This is converted into text (feature contributions) and passed to **Gemini LLM** for a **natural language explanation**.

---

## 6. Final API Output

The Flask backend returns JSON like this:

```json
{
  "cbsc_score": 72.5,
  "risk_category": "Low Risk",
  "explanation": "The customer has a high repayment ratio and timely bill payments, which reduce risk. The outstanding loan is moderate but manageable."
}
```

---
Great question 🙌 — let’s expand the doc with **how XGBoost actually predicts the credit score** in plain English and technical terms.

---


## 7. How XGBoost Predicts the Credit Score

XGBoost is a **gradient boosting algorithm**. It works by building an **ensemble of decision trees**, where each new tree learns to **correct the mistakes** made by the previous trees.

### 🔹 Step-by-Step (Plain English)

1. **Initial Guess**:

   * The model starts with a baseline prediction (often the average CBSC score of the training data).

2. **Tree Building**:

   * XGBoost builds small regression trees.
   * Each tree looks at **patterns in the features** (like repayment ratio, outstanding amount, utility bill history) and learns how they influence the CBSC score.

3. **Boosting Process**:

   * After the first tree, the model checks the **errors (residuals)** between predicted score and actual score.
   * The next tree is trained specifically to predict those errors.
   * This process repeats for 100+ trees (depending on `n_estimators`).

4. **Final Prediction**:

   * All the trees’ outputs are combined (weighted sum) to generate the final CBSC score for a customer.

   Example:

   ```
   CBSC_pred = Tree1 + Tree2 + Tree3 + ... + Tree100
   ```

5. **Why It Works Well**:

   * XGBoost handles non-linear relationships between features and credit score.
   * It automatically figures out **interactions** (e.g., “High loan amount” + “Low repayment ratio” → high risk).
   * It gives more importance to the features that strongly impact the score.

---

### 🔹 Example with a Customer

Suppose a beneficiary has:

* Loan Amount = 20,000
* Repayment Ratio = 0.85
* Utility Bill Timeliness = 0.90
* Outstanding Amount = 2,000

XGBoost will:

1. Start with an average score (say 65).
2. Add adjustments from each tree:

   * Tree 1: +5 (because high repayment ratio → reliable)
   * Tree 2: -3 (because loan amount is high)
   * Tree 3: +4 (utility bills are mostly on time)
   * … (continues for 100 trees)
3. Final Score ≈ 71 → categorized as **Low Risk**.

---

### 🔹 SHAP for Transparency

While XGBoost gives the **final score**, SHAP breaks it down into **contributions from each feature**.

Example explanation:

* `repayments_on_time_ratio` contributed **+8 points**
* `outstanding_amount` contributed **-5 points**
* `utility_bill_on_time_ratio` contributed **+3 points**

So the model is not a “black box” — we can explain *why* it gave a certain score.

---

✅ This way, the backend not only predicts the **CBSC score** but also justifies it using SHAP.

---


## 7. Why We Used XGBoost

XGBoost (**Extreme Gradient Boosting**) is a powerful algorithm for both classification and regression tasks. For our **Beneficiary Credit Scoring (CBSC)** problem, it was chosen because of the following strengths:

---

### 🔹 1. Handles Missing (Null) Values Gracefully

* In real-world beneficiary data, some records may have missing values (e.g., no utility bill data, incomplete repayment history).
* XGBoost has a built-in mechanism to handle missing values:

  * While training, it **learns the best default direction** to take whenever a feature value is missing.
  * During prediction, if a value is null, XGBoost automatically follows that “default path” instead of failing.
* This makes it **robust in real-world digital lending scenarios**, where data is often incomplete.

---

### 🔹 2. High Predictive Accuracy

* XGBoost uses **gradient boosting** over decision trees, which combines the strengths of multiple weak learners (trees).
* It can capture complex **non-linear relationships** between financial features and repayment ability better than simpler models like Logistic Regression or Linear Regression.
* It consistently performs better in **tabular data problems** (like credit scoring) compared to many other ML algorithms.

---

### 🔹 3. Feature Importance & Explainability (with SHAP)

* XGBoost allows us to extract **feature importance** directly.
* Combined with SHAP values, we can clearly explain *which factors increased or decreased a beneficiary’s credit score*.
* This transparency is critical for **compliance and audits** in concessional lending.

---

### 🔹 4. Handles Both Small and Large Datasets

* For prototyping (like our fake dataset of 50 beneficiaries), it works well without heavy tuning.
* For large-scale production data (thousands of records), XGBoost is optimized for **speed and efficiency** with parallel training.

---

### 🔹 5. Flexibility in Output

* XGBoost can output either:

  * **Probabilities / classifications** (e.g., “Low Risk” vs. “High Risk”), or
  * **Continuous scores** (like our CBSC score from 0–100).
* This dual nature makes it **perfect for credit scoring**, where we want both a **score** and a **risk band**.

---

### 🔹 6. Proven Track Record

* XGBoost has been the **winning algorithm** in many Kaggle competitions involving structured/tabular financial data.
* It’s widely trusted in industry applications like fraud detection, churn prediction, and **credit scoring**.

---

✅ In short:
We used **XGBoost** because it is **accurate, explainable, robust to missing values, and well-suited for tabular beneficiary data**.

---


✅ This pipeline ensures that the system can **predict creditworthiness, explain decisions, and provide interpretable results** to the frontend.

---


