# ML automation work

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

# Data ingestion

def load_data():
    "Simualtes loading data from a CSV or database."
    print("[1/5] loading data...")
    # generate simple binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=42)
    feature_names = [f"feature{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df

# Automated ML pipeline setup

def build_pipeline():
    " Creates an end-to-end scikit-learn pipeline."
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    return pipeline

# Training and evaluation

def run_pipeline():
    # step 1: load
    df = load_data()

    X = df.drop(columns=['target'])
    y = df['target']

    # step 2: train/test split
    print("[2/5] splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # step 3: train pipeline
    print("[3/5] preprocessing and training model...")
    model = build_pipeline()
    model.fit(X_train, y_train)

    # step 4: evaluate
    print("[4/5] evaluating model performance...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Accuracy: {accuracy * 100: .2f}% ---")
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred))

    # step 5: save artifact
    print("[5/5] savingmodel pipeline artifact...")
    joblib.dump(model, 'ml_model_pipeline.pkl')
    print("Pipeline successfully saved to 'ml_model_pipeline.pkl' !\n")

if __name__ == "__main__":
    run_pipeline()