import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, average_precision_score, roc_auc_score)
from sklearn.pipeline import Pipeline
import joblib

# data ingestion
def load_datasets(train_path="fraudTrain.csv", test_path="fraudTest.csv"):
    for path in [train_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find '{path}'. Please ensure both 'fraudTrain.csv' "
            f"and 'fraudTest.csv' are in your 'ML work' folder.")
    print(f"[1/5] loading '{train_path}' and '{test_path}'...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f" training transactions: {len(train_df):,}")
    print(f" testing transactions: {len(test_df):,}")
    return train_df, test_df

# feature selection & preprocessing pipeline
def build_pipeline():
    "builds a pipeline capable of handling both numeric and categorical columns"
    numeric_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
    categorical_features = ['category']

    preprocessor = ColumnTransformer(transformers = [('num', RobustScaler(), numeric_features), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1))])
    return pipeline, numeric_features + categorical_features


# training and evaluation
def run_fraud_pipeline():
    # step 1: load both files
    train_df, test_df = load_datasets("fraudTrain.csv", "fraudTest.csv")

    # step 2: build pipeline
    pipeline, features = build_pipeline()
    X_train = train_df[features]
    y_train = train_df['is_fraud']
    X_test = test_df[features]
    y_test = test_df['is_fraud']

    # step 3: train model pipeline
    print("[2/5] preprocessing and training RandomForest Classifier...")
    pipeline.fit(X_train, y_train)

    # step 4: evalaute model on test set
    print("[3/5] evalauting fraud model on 'fraudTest.csv'...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "="*50)
    print(" evaluation results")
    print("="*50)
    print(f"Area under precision-recall curve (pr-auc): {pr_auc:.4f}")
    print(f"Area under roc curve (roc-auc): {roc_auc:.4f}")
    print("\nConfusion matrix:")
    print(f" true negatives (legit detected as legit): {cm[0][0]:>8}")
    print(f" false positives (legit flagged as fraud): {cm[0][1]:>8}")
    print(f" false negatives (fraud missed by model): {cm[1][0]:>8}")
    print(f" true positives (fraud correctly caught): {cm[1][1]:>8}")

    print("\nClassification Report: ")
    print(classification_report(y_test, y_pred, digits=4))

    # step 5: save model
    artifact_name = 'fraud_detection_pipeline.pkl'
    print(f"[5/5] saving model pipeline to '{artifact_name}'...")
    joblib.dump(pipeline, artifact_name)
    print("pipeline successfully saved\n")

if __name__ == "__main__":
    run_fraud_pipeline()