"""
Loan Eligibility Model Training Script
Trains a RandomForest classifier pipeline and saves it for inference
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib 
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic loan application data if CSV not found
    """
    np.random.seed(42)
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
        'ApplicantIncome': np.random.randint(1000, 15000, n_samples),
        'CoapplicantIncome': np.random.randint(0, 8000, n_samples),
        'LoanAmount': np.random.randint(50, 500, n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 240, 120], n_samples),
        'Credit_History': np.random.choice([0.0, 1.0], n_samples, p=[0.15, 0.85]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target based on realistic logic
    df['Loan_Status'] = 'N'
    
    # Higher approval for good credit, higher income, and graduates
    approval_prob = (
        df['Credit_History'] * 0.6 +
        (df['ApplicantIncome'] > 5000).astype(int) * 0.2 +
        (df['Education'] == 'Graduate').astype(int) * 0.1 +
        (df['LoanAmount'] < 200).astype(int) * 0.1
    )
    
    df.loc[approval_prob > 0.5, 'Loan_Status'] = 'Y'
    
    # Add some randomness
    random_flips = np.random.choice(df.index, size=int(0.1 * n_samples), replace=False)
    df.loc[random_flips, 'Loan_Status'] = df.loc[random_flips, 'Loan_Status'].map({'Y': 'N', 'N': 'Y'})
    
    # Introduce some missing values
    missing_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
    df.loc[missing_indices, 'Credit_History'] = np.nan
    
    return df


def load_data():
    """
    Load training data from CSV or generate synthetic data
    """
    data_path = 'data/loan_train.csv'

    if os.path.exists(data_path):
        print(f"✓ Loading real dataset from {data_path}")
        df = pd.read_csv(data_path)
        # print(df.head())  corrently loading data
        df["__data_source__"] = "real"
    else:
        print("✗ Dataset not found. Generating synthetic data...")
        df = generate_synthetic_data(1000)
        df["__data_source__"] = "synthetic"
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"✓ Synthetic dataset saved to {data_path}")
        # print(df.head())


    return df



def engineer_features(df):
    """
    Create additional features for better predictions
    """
    df = df.copy()
    
    # Total household income
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    
    # Income to loan ratio
    df['IncomeToLoanRatio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)  # +1 to avoid division by zero
    
    # Log transform of loan amount (reduces skewness)
    df['LogLoanAmount'] = np.log1p(df['LoanAmount'])
    
    # Debt to income ratio
    df['DebtToIncomeRatio'] = df['LoanAmount'] / (df['TotalIncome'] + 1)
    
    # EMI calculation (approximate monthly payment)
    df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] / 12 + 1)
    
    # Balance after EMI
    df['BalanceIncome'] = df['TotalIncome'] - df['EMI']
    
    return df


def build_pipeline():
    """
    Build preprocessing and modeling pipeline
    """
    # Define feature groups
    numeric_features = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
        'Loan_Amount_Term', 'Credit_History',
        'TotalIncome', 'IncomeToLoanRatio', 'LogLoanAmount',
        'DebtToIncomeRatio', 'EMI', 'BalanceIncome'
    ]
    
    categorical_features = [
        'Gender', 'Married', 'Dependents', 'Education', 
        'Self_Employed', 'Property_Area'
    ]
    
    # Numeric transformer: impute with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer: impute with 'Unknown', then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Full pipeline with model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return pipeline


def evaluate_model(y_true, y_pred, y_proba):
    """
    Print comprehensive evaluation metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    
    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label='Y'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, pos_label='Y'):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, pos_label='Y'):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Approved', 'Approved']))
    print("="*60 + "\n")


def main():
    """
    Main training pipeline
    """
    print("\n🚀 Starting Loan Eligibility Model Training...\n")
    
    # Load data
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Loan_Status'].value_counts()}\n")
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare features and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}\n")
    
    # Build and train pipeline
    print("🔧 Building ML pipeline...")
    pipeline = build_pipeline()
    
    print("🎯 Training model...")
    pipeline.fit(X_train, y_train)
    print("✓ Training complete!\n")
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred, y_proba)
    
    # Save model
    model_dir = 'backend/models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.joblib')
    
    joblib.dump(pipeline, model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Feature importance
    feature_names = (
        ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
         'Loan_Amount_Term', 'Credit_History', 'TotalIncome', 
         'IncomeToLoanRatio', 'LogLoanAmount', 'DebtToIncomeRatio', 
         'EMI', 'BalanceIncome'] +
        list(pipeline.named_steps['preprocessor']
             .named_transformers_['cat']
             .named_steps['onehot']
             .get_feature_names_out())
    )
    
    importances = pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))
    
    print("\n✅ Training pipeline completed successfully!\n")


if __name__ == "__main__":
    main()