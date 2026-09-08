import sys
import os
import joblib
sys.path.insert(0, '.')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

from src.preprocessing import load_and_clean, build_pipeline
from src.features import add_features


def main():
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean('data/telco_churn.csv')

    # Add engineered features
    print("Adding engineered features...")
    df = add_features(df)

    # Split into X and y
    print("Splitting data...")
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Train/test split: 80/20, stratified, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.4f}")

    # Create XGBClassifier with specified parameters
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05
    )

    # Build pipeline with the configured model
    print("Building pipeline...")
    pipeline = build_pipeline(xgb_model)

    # Fit the pipeline
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Save the pipeline
    print("\n" + "="*60)
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'churn_pipeline.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()
