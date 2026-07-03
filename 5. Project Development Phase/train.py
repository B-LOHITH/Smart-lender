import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

def main():
    print("--- Step 1: Loading Dataset ---")
    dataset_path = os.path.join("dataset", "loan_prediction.csv")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())

    print("\n--- Step 2: Data Preprocessing ---")
    
    # 1. Remove duplicate rows if any
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        print(f"Removing {duplicates_count} duplicate rows.")
        df = df.drop_duplicates()
        
    # Remove Loan_ID as it is unique and not useful for modeling
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    # 2. Fill Missing Values
    # Categorical Columns (Mode)
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    modes = {}
    for col in cat_cols:
        mode_val = df[col].mode()[0]
        modes[col] = mode_val
        df[col] = df[col].fillna(mode_val)
        print(f"Imputed missing values in '{col}' with Mode: {mode_val}")

    # Numerical Columns
    # LoanAmount (Median)
    loan_amount_median = df['LoanAmount'].median()
    df['LoanAmount'] = df['LoanAmount'].fillna(loan_amount_median)
    print(f"Imputed missing values in 'LoanAmount' with Median: {loan_amount_median}")

    # Loan_Amount_Term (Mode - typically 360.0)
    loan_term_mode = df['Loan_Amount_Term'].mode()[0]
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(loan_term_mode)
    print(f"Imputed missing values in 'Loan_Amount_Term' with Mode: {loan_term_mode}")

    # 3. Feature Encoding
    gender_map = {'Male': 1, 'Female': 0}
    married_map = {'Yes': 1, 'No': 0}
    dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
    education_map = {'Graduate': 1, 'Not Graduate': 0}
    self_employed_map = {'Yes': 1, 'No': 0}
    property_area_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    loan_status_map = {'Y': 1, 'N': 0}

    df['Gender'] = df['Gender'].map(gender_map)
    df['Married'] = df['Married'].map(married_map)
    df['Dependents'] = df['Dependents'].map(dependents_map)
    df['Education'] = df['Education'].map(education_map)
    df['Self_Employed'] = df['Self_Employed'].map(self_employed_map)
    df['Property_Area'] = df['Property_Area'].map(property_area_map)
    df['Loan_Status'] = df['Loan_Status'].map(loan_status_map)

    print("\nDataset after Encoding:")
    print(df.head())

    # 4. Train-Test Split
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Training target class balance:\n{y_train.value_counts(normalize=True)}")

    # 5. Handle Imbalance using SMOTE (on training set only)
    print("\nApplying SMOTE to balance class distribution...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Resampled training set shape: {X_train_res.shape}")
    print(f"Resampled target class balance:\n{y_train_res.value_counts(normalize=True)}")

    # 6. Normalize Features using StandardScaler
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Save preprocessing elements (scaler, modes, medians, maps)
    preprocess_metadata = {
        'modes': modes,
        'loan_amount_median': loan_amount_median,
        'loan_term_mode': loan_term_mode,
        'gender_map': gender_map,
        'married_map': married_map,
        'dependents_map': dependents_map,
        'education_map': education_map,
        'self_employed_map': self_employed_map,
        'property_area_map': property_area_map,
        'feature_names': list(X.columns)
    }

    # 7. Model Building & Evaluation
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': xgb.XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss')
    }

    best_model_name = None
    best_accuracy = 0
    best_model = None
    
    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_scaled, y_train_res)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'Confusion Matrix': cm.tolist()
        }
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # We prefer XGBoost as final model as per requirements, but let's track the best one too
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

    # Select final model: We explicitly choose XGBoost as per requirements, 
    # but if XGBoost isn't the best, we still save XGBoost as instructed ("Chosen as final model.").
    # Let's save XGBoost as the final model.
    final_model = models['XGBoost']
    print(f"\nSaving final model (XGBoost) and Scaler to disk...")
    
    # Save the scaler, preprocessing metadata, and the trained model
    with open('model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
        
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open('preprocess_metadata.pkl', 'wb') as f:
        pickle.dump(preprocess_metadata, f)
        
    print("Files saved successfully: model.pkl, scaler.pkl, preprocess_metadata.pkl")

if __name__ == "__main__":
    main()
