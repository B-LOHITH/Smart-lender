from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Paths for the saved models and metadata
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'
METADATA_PATH = 'preprocess_metadata.pkl'

model = None
scaler = None
metadata = None

def load_assets():
    global model, scaler, metadata
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(METADATA_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            with open(METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)
            print("Model and preprocessing assets loaded successfully!")
            return True
        else:
            print("Warning: Model files not found. Please run train.py first.")
            return False
    except Exception as e:
        print(f"Error loading model files: {e}")
        return False

# Initialize assets
load_assets()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model, scaler, metadata
    
    # If model is not loaded yet, try to load it
    if model is None or scaler is None:
        if not load_assets():
            return "Model is not trained yet. Please run the training script (train.py) first to generate model.pkl, scaler.pkl, and preprocess_metadata.pkl.", 503

    if request.method == 'GET':
        return render_template('predict.html')
    
    elif request.method == 'POST':
        try:
            # 1. Retrieve inputs from the form
            gender = request.form.get('Gender', 'Male')
            married = request.form.get('Married', 'No')
            dependents = request.form.get('Dependents', '0')
            education = request.form.get('Education', 'Graduate')
            self_employed = request.form.get('Self_Employed', 'No')
            
            try:
                applicant_income = float(request.form.get('ApplicantIncome', 0))
            except ValueError:
                applicant_income = 0.0
                
            try:
                coapplicant_income = float(request.form.get('CoapplicantIncome', 0))
            except ValueError:
                coapplicant_income = 0.0
                
            try:
                loan_amount = float(request.form.get('LoanAmount', 0))
            except ValueError:
                loan_amount = 0.0
                
            try:
                loan_term = float(request.form.get('Loan_Amount_Term', 360))
            except ValueError:
                loan_term = 360.0
                
            try:
                credit_history = float(request.form.get('Credit_History', 1))
            except ValueError:
                credit_history = 1.0
                
            property_area = request.form.get('Property_Area', 'Semiurban')

            # 2. Map form string inputs to numerical codes using the training maps
            gender_encoded = metadata['gender_map'].get(gender, 1)
            married_encoded = metadata['married_map'].get(married, 0)
            dependents_encoded = metadata['dependents_map'].get(dependents, 0)
            education_encoded = metadata['education_map'].get(education, 1)
            self_employed_encoded = metadata['self_employed_map'].get(self_employed, 0)
            property_area_encoded = metadata['property_area_map'].get(property_area, 1)

            # Assemble features in the correct order:
            # ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
            #  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            #  'Credit_History', 'Property_Area']
            features = [
                gender_encoded,
                married_encoded,
                dependents_encoded,
                education_encoded,
                self_employed_encoded,
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_term,
                credit_history,
                property_area_encoded
            ]

            # Convert to DataFrame with correct feature names to match XGBoost/Scaler
            features_df = pd.DataFrame([features], columns=metadata['feature_names'])

            # 3. Scale the features using the loaded StandardScaler
            features_scaled = scaler.transform(features_df)

            # 4. Predict
            prediction = int(model.predict(features_scaled)[0])
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = round(float(probabilities[prediction]) * 100, 2)

            result_text = "Approved" if prediction == 1 else "Rejected"

            # Create a dictionary of input details to display back on the submit page
            input_details = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self Employed': self_employed,
                'Applicant Income': f"${applicant_income:,.2f}",
                'Coapplicant Income': f"${coapplicant_income:,.2f}",
                'Total Monthly Income': f"${(applicant_income + coapplicant_income):,.2f}",
                'Requested Loan Amount': f"${(loan_amount * 1000):,.2f}",
                'Loan Term (Months)': int(loan_term) if loan_term % 1 == 0 else loan_term,
                'Credit History': "Good / Clean Record" if credit_history == 1.0 else "Poor Credit / Default Record",
                'Property Area': property_area
            }

            return render_template('submit.html', 
                                   prediction=result_text, 
                                   confidence=confidence, 
                                   input_details=input_details)
                                   
        except Exception as e:
            return f"An error occurred during prediction: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
