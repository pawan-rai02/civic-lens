# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Install required libraries
# MAGIC %pip install xgboost shap scikit-learn==1.5.2 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load data and EDA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
import shap
import matplotlib.pyplot as plt

# Load the Gold training table with sampling for memory efficiency
print("Loading civic_lens.ml.nyc_training...")
# Sample 10% of data (stratified by target) for faster training
sample_fraction = 0.1
df_spark = spark.table("civic_lens.ml.nyc_training")
total_rows = df_spark.count()
print(f"Total rows: {total_rows:,}")
print(f"Sampling {sample_fraction*100}% for training...")

df = df_spark.sample(fraction=sample_fraction, seed=42).toPandas()
print(f"✓ Loaded {len(df):,} sampled rows, {len(df.columns)} columns")

# Display basic info
print(f"\nTarget distributions:")
display(df['resolution_days'].describe())
print(f"\nnever_resolved class balance:")
display(df['never_resolved'].value_counts(normalize=True))

# COMMAND ----------

# DBTITLE 1,Feature selection and data splitting
# Define feature columns (exclude identifiers and targets)
feature_cols = [
    'dow_filed', 'hour_filed', 'month_filed',
    'latitude', 'longitude',
    'complaint_type_enc', 'agency_enc',
    'borough_blackhole_rate', 'agency_resolution_rate_hist', 'agency_open_complaints_30d',
    'urgency_score', 'topic_id'
] + [f'tfidf_feat_{i}' for i in range(1, 51)]

print(f"Using {len(feature_cols)} features")

# Separate regression and classification datasets
# For regression: only rows with non-null resolution_days (resolved complaints)
df_reg = df[df['resolution_days'].notna()].copy()
X_reg = df_reg[feature_cols]
y_reg = df_reg['resolution_days']

# For classification: all rows
X_clf = df[feature_cols]
y_clf = df['never_resolved']

print(f"\nRegression dataset: {len(X_reg):,} rows")
print(f"Classification dataset: {len(X_clf):,} rows")
print(f"Class balance: {y_clf.value_counts(normalize=True).to_dict()}")

# Split data (70% train, 15% val, 15% test)
X_reg_train, X_reg_temp, y_reg_train, y_reg_temp = train_test_split(
    X_reg, y_reg, test_size=0.30, random_state=42
)
X_reg_val, X_reg_test, y_reg_val, y_reg_test = train_test_split(
    X_reg_temp, y_reg_temp, test_size=0.50, random_state=42
)

X_clf_train, X_clf_temp, y_clf_train, y_clf_temp = train_test_split(
    X_clf, y_clf, test_size=0.30, random_state=42, stratify=y_clf
)
X_clf_val, X_clf_test, y_clf_val, y_clf_test = train_test_split(
    X_clf_temp, y_clf_temp, test_size=0.50, random_state=42, stratify=y_clf_temp
)

print(f"\nRegression splits:")
print(f"  Train: {len(X_reg_train):,}, Val: {len(X_reg_val):,}, Test: {len(X_reg_test):,}")
print(f"\nClassification splits:")
print(f"  Train: {len(X_clf_train):,}, Val: {len(X_clf_val):,}, Test: {len(X_clf_test):,}")

# COMMAND ----------

# DBTITLE 1,Preprocess features
# Handle missing values and scale features
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# Fit on regression training data
X_reg_train_imputed = imputer.fit_transform(X_reg_train)
X_reg_train_scaled = scaler.fit_transform(X_reg_train_imputed)

# Transform validation and test sets
X_reg_val_scaled = scaler.transform(imputer.transform(X_reg_val))
X_reg_test_scaled = scaler.transform(imputer.transform(X_reg_test))

# For classification (fit on classification training data)
imputer_clf = SimpleImputer(strategy='mean')
scaler_clf = StandardScaler()

X_clf_train_imputed = imputer_clf.fit_transform(X_clf_train)
X_clf_train_scaled = scaler_clf.fit_transform(X_clf_train_imputed)

X_clf_val_scaled = scaler_clf.transform(imputer_clf.transform(X_clf_val))
X_clf_test_scaled = scaler_clf.transform(imputer_clf.transform(X_clf_test))

print("✓ Features imputed and scaled")

# COMMAND ----------

# DBTITLE 1,Train all 4 models with MLflow tracking
# Set MLflow experiment
mlflow.set_experiment("/Users/pawanvirat32@gmail.com/civic-lens-experiments")

# Dictionary to store best models and their metrics
model_results = {}

print("="*80)
print("TRAINING REGRESSION MODELS")
print("="*80)

# 1. XGBoost Regressor
print("\n1. Training XGBoost Regressor...")
with mlflow.start_run(run_name="xgboost_regressor") as run:
    mlflow.set_tag("model_type", "regression")
    mlflow.set_tag("algorithm", "xgboost")
    mlflow.set_tag("target", "resolution_days")
    
    xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_reg.fit(X_reg_train_scaled, y_reg_train)
    
    y_pred_test = xgb_reg.predict(X_reg_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_test))
    test_mae = mean_absolute_error(y_reg_test, y_pred_test)
    test_r2 = r2_score(y_reg_test, y_pred_test)
    
    mlflow.log_params({"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1})
    mlflow.log_metrics({"test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2})
    
    signature = infer_signature(X_reg_test_scaled, y_pred_test)
    mlflow.xgboost.log_model(xgb_reg, "model", signature=signature, input_example=X_reg_test_scaled[:5])
    
    model_results['xgb_reg'] = {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2, 'run_id': run.info.run_id}
    print(f"  ✓ Test RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")

# 2. Random Forest Regressor
print("\n2. Training Random Forest Regressor...")
with mlflow.start_run(run_name="random_forest_regressor") as run:
    mlflow.set_tag("model_type", "regression")
    mlflow.set_tag("algorithm", "random_forest")
    mlflow.set_tag("target", "resolution_days")
    
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_reg.fit(X_reg_train_scaled, y_reg_train)
    
    y_pred_test = rf_reg.predict(X_reg_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_test))
    test_mae = mean_absolute_error(y_reg_test, y_pred_test)
    test_r2 = r2_score(y_reg_test, y_pred_test)
    
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})
    mlflow.log_metrics({"test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2})
    
    signature = infer_signature(X_reg_test_scaled, y_pred_test)
    mlflow.sklearn.log_model(rf_reg, "model", signature=signature, input_example=X_reg_test_scaled[:5])
    
    model_results['rf_reg'] = {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2, 'run_id': run.info.run_id}
    print(f"  ✓ Test RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")

print("\n" + "="*80)
print("TRAINING CLASSIFICATION MODELS")
print("="*80)

# 3. XGBoost Classifier
print("\n3. Training XGBoost Classifier...")
with mlflow.start_run(run_name="xgboost_classifier") as run:
    mlflow.set_tag("model_type", "classification")
    mlflow.set_tag("algorithm", "xgboost")
    mlflow.set_tag("target", "never_resolved")
    
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_clf.fit(X_clf_train_scaled, y_clf_train)
    
    y_pred_test = xgb_clf.predict(X_clf_test_scaled)
    y_pred_proba = xgb_clf.predict_proba(X_clf_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_clf_test, y_pred_test)
    test_precision = precision_score(y_clf_test, y_pred_test)
    test_recall = recall_score(y_clf_test, y_pred_test)
    test_f1 = f1_score(y_clf_test, y_pred_test)
    test_auc = roc_auc_score(y_clf_test, y_pred_proba)
    
    mlflow.log_params({"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1})
    mlflow.log_metrics({"test_accuracy": test_acc, "test_precision": test_precision, "test_recall": test_recall, "test_f1": test_f1, "test_auc": test_auc})
    
    signature = infer_signature(X_clf_test_scaled, y_pred_proba)
    mlflow.xgboost.log_model(xgb_clf, "model", signature=signature, input_example=X_clf_test_scaled[:5])
    
    model_results['xgb_clf'] = {'acc': test_acc, 'precision': test_precision, 'recall': test_recall, 'f1': test_f1, 'auc': test_auc, 'run_id': run.info.run_id}
    print(f"  ✓ Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

# 4. Logistic Regression
print("\n4. Training Logistic Regression...")
with mlflow.start_run(run_name="logistic_regression") as run:
    mlflow.set_tag("model_type", "classification")
    mlflow.set_tag("algorithm", "logistic_regression")
    mlflow.set_tag("target", "never_resolved")
    
    lr_clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr_clf.fit(X_clf_train_scaled, y_clf_train)
    
    y_pred_test = lr_clf.predict(X_clf_test_scaled)
    y_pred_proba = lr_clf.predict_proba(X_clf_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_clf_test, y_pred_test)
    test_precision = precision_score(y_clf_test, y_pred_test)
    test_recall = recall_score(y_clf_test, y_pred_test)
    test_f1 = f1_score(y_clf_test, y_pred_test)
    test_auc = roc_auc_score(y_clf_test, y_pred_proba)
    
    mlflow.log_params({"max_iter": 1000})
    mlflow.log_metrics({"test_accuracy": test_acc, "test_precision": test_precision, "test_recall": test_recall, "test_f1": test_f1, "test_auc": test_auc})
    
    signature = infer_signature(X_clf_test_scaled, y_pred_proba)
    mlflow.sklearn.log_model(lr_clf, "model", signature=signature, input_example=X_clf_test_scaled[:5])
    
    model_results['lr_clf'] = {'acc': test_acc, 'precision': test_precision, 'recall': test_recall, 'f1': test_f1, 'auc': test_auc, 'run_id': run.info.run_id}
    print(f"  ✓ Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE")
print("="*80)

# COMMAND ----------

# DBTITLE 1,Register champion models to Unity Catalog
print("\n" + "="*80)
print("REGISTERING CHAMPION MODELS TO UNITY CATALOG")
print("="*80)

# Determine regression champion (lower RMSE is better)
reg_champion = 'xgb_reg' if model_results['xgb_reg']['rmse'] < model_results['rf_reg']['rmse'] else 'rf_reg'
reg_champion_name = "XGBoost" if reg_champion == 'xgb_reg' else "Random Forest"
reg_champion_run_id = model_results[reg_champion]['run_id']

print(f"\nRegression Champion: {reg_champion_name}")
print(f"  RMSE: {model_results[reg_champion]['rmse']:.2f}")
print(f"  MAE: {model_results[reg_champion]['mae']:.2f}")
print(f"  R²: {model_results[reg_champion]['r2']:.4f}")

# Register regression champion
model_uri = f"runs:/{reg_champion_run_id}/model"
registered_reg_model = mlflow.register_model(
    model_uri=model_uri,
    name="civic_lens.ml.nyc_resolution_regressor"
)
print(f"  ✓ Registered as civic_lens.ml.nyc_resolution_regressor (version {registered_reg_model.version})")

# Determine classification champion (higher AUC is better)
clf_champion = 'xgb_clf' if model_results['xgb_clf']['auc'] > model_results['lr_clf']['auc'] else 'lr_clf'
clf_champion_name = "XGBoost" if clf_champion == 'xgb_clf' else "Logistic Regression"
clf_champion_run_id = model_results[clf_champion]['run_id']

print(f"\nClassification Champion: {clf_champion_name}")
print(f"  Accuracy: {model_results[clf_champion]['acc']:.4f}")
print(f"  Precision: {model_results[clf_champion]['precision']:.4f}")
print(f"  Recall: {model_results[clf_champion]['recall']:.4f}")
print(f"  F1: {model_results[clf_champion]['f1']:.4f}")
print(f"  AUC: {model_results[clf_champion]['auc']:.4f}")

# Register classification champion
model_uri = f"runs:/{clf_champion_run_id}/model"
registered_clf_model = mlflow.register_model(
    model_uri=model_uri,
    name="civic_lens.ml.nyc_blackhole_classifier"
)
print(f"  ✓ Registered as civic_lens.ml.nyc_blackhole_classifier (version {registered_clf_model.version})")

print("\n" + "="*80)
print("✓ MODEL REGISTRATION COMPLETE")
print("="*80)
print(f"\nRegistered Models:")
print(f"  1. civic_lens.ml.nyc_resolution_regressor v{registered_reg_model.version} ({reg_champion_name})")
print(f"  2. civic_lens.ml.nyc_blackhole_classifier v{registered_clf_model.version} ({clf_champion_name})")
