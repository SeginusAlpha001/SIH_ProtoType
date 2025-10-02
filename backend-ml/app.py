import shap
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import google.generativeai as genai
from pymongo import MongoClient
import os

from dotenv import load_dotenv
import os

load_dotenv()

# Load saved model, scaler, and feature names
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

app = Flask(__name__)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
client = MongoClient(os.environ.get("MONGODB_URI"))
db = client['your_db_name']
users_collection = db['users']

LLM_MODEL = "gemini-1.5-pro"
llm = genai.GenerativeModel(model_name=LLM_MODEL)

def classify_risk(score):
    if score >= 80:
        return "Low Risk"
    elif 60 <= score < 80:
        return "Medium Risk"
    else:
        return "High Risk"

@app.route('/new', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.json  

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # One-hot encode
        df = pd.get_dummies(df, drop_first=True)

        # Align with training columns
        df = df.reindex(columns=feature_names, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df)

        # Predict CBSC
        score = model.predict(df_scaled)[0]
        risk_band = classify_risk(score)

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_scaled)

        # Get feature importance for this instance
        feature_importance = {
            feature: float(value)
            for feature, value in zip(feature_names, shap_values[0])
        }
        shap_text = ", ".join([f"{k}: {v:.3f}" for k, v in feature_importance.items()])
        prompt = f"Act as a bank employee which analyses the finacial credentials of a customer and then can tell where this person is realible to give a loan, you are given all the values, analyze them and tell wether this person fits for a loan or not and also why : {shap_text}"

        # 8. Call Gemini API
        response = llm.generate_content(prompt)

        # Extract natural explanation
        natural_explanation = response.text
        # Build response
        response = {
            "score": float(score),
            "risk_band": risk_band,
            "explanation": natural_explanation
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/update', methods=['POST'])
def update_predict():
    try:
        # 1. Get JSON data
        data = request.json
        user_id = data.get("beneficiary_id")  # or email

        if not user_id:
            return jsonify({"error": "beneficiary_id is required"})

        # 2. Fetch existing user data from MongoDB
        user_record = users_collection.find_one({"beneficiary_id": user_id})
        if not user_record:
            return jsonify({"error": "User not found"})

        # 3. Update only the provided attributes
        # Example: repayments_on_time and/or num_past_loans
        for key in ["repayments_on_time", "num_past_loans"]:
            if key in data:
                user_record[key] = data[key]

        # 4. Remove MongoDB internal fields
        user_record.pop("_id", None)
        df = pd.DataFrame([data])

        # One-hot encode
        df = pd.get_dummies(df, drop_first=True)

        # Align with training columns
        df = df.reindex(columns=feature_names, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df)

        # Predict CBSC
        score = model.predict(df_scaled)[0]
        risk_band = classify_risk(score)

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_scaled)

        # Get feature importance for this instance
        feature_importance = {
            feature: float(value)
            for feature, value in zip(feature_names, shap_values[0])
        }
        shap_text = ", ".join([f"{k}: {v:.3f}" for k, v in feature_importance.items()])
        prompt = f"Act as a bank employee which analyses the financial credentials of a customer and then can tell whether this person is reliable to give a loan, you are given all the values, analyze them and tell whether this person fits for a loan or not and also why : {shap_text}"

        response = llm.generate_content(prompt)

        # Extract natural explanation
        natural_explanation = response.text

        # Build response
        response = {
            "score": float(score),
            "risk_band": risk_band,
            "explanation": natural_explanation
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)