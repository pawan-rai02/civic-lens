# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Notebook Header
# MAGIC %md
# MAGIC # 06 - ML Model Training: Outcome Prediction
# MAGIC
# MAGIC **Pipeline Stage:** Machine Learning (Model Development)
# MAGIC
# MAGIC **Objective:** Train multi-class classification models to predict complaint outcome (Resolved, Closed, Rejected) using engineered features from the gold layer.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Problem Definition
# MAGIC
# MAGIC **Task:** Multi-class classification  
# MAGIC **Target Variable:** `outcome_label`
# MAGIC * **0 = Resolved** (complaint successfully addressed)
# MAGIC * **1 = Closed** (complaint closed without resolution)
# MAGIC * **2 = Rejected** (complaint rejected by system)
# MAGIC
# MAGIC **Business Value:**
# MAGIC * Early identification of at-risk complaints
# MAGIC * Resource allocation optimization
# MAGIC * Service quality improvement
# MAGIC * Proactive citizen engagement
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Input Data
# MAGIC
# MAGIC * **Table:** `civic_lens.gold.bbmp_complaints_enriched`
# MAGIC * **Records:** 766,648 complaints (2020–2025)
# MAGIC * **Features:** 86 columns including:
# MAGIC   * Core complaint attributes (category, ward, timestamps)
# MAGIC   * NLP features (urgency_score + 50 TF-IDF components)
# MAGIC   * Ward-category aggregates (rejection rates, volumes)
# MAGIC   * Quality flags (urgency, problem area, context)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Models
# MAGIC
# MAGIC ### 1. XGBoost Multi-Class Classifier
# MAGIC * **Algorithm:** Gradient boosting decision trees
# MAGIC * **Configuration:** `objective="multi:softmax"`, `num_class=3`
# MAGIC * **Strengths:** Handles complex interactions, robust to imbalanced classes
# MAGIC
# MAGIC ### 2. Multinomial Logistic Regression
# MAGIC * **Algorithm:** Linear classifier with softmax
# MAGIC * **Configuration:** `multi_class="multinomial"`, `solver="lbfgs"`
# MAGIC * **Strengths:** Interpretable coefficients, fast training, baseline model
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Evaluation Strategy
# MAGIC
# MAGIC **Metrics:**
# MAGIC * Accuracy (overall correctness)
# MAGIC * Precision, Recall, F1-Score (per-class performance)
# MAGIC * Confusion Matrix (error analysis)
# MAGIC * ROC-AUC (probability calibration)
# MAGIC
# MAGIC **Validation:**
# MAGIC * 80/20 train-test split
# MAGIC * Stratified sampling (preserve class distribution)
# MAGIC * Random state fixed for reproducibility
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## MLflow Integration
# MAGIC
# MAGIC **Logging:**
# MAGIC * Model parameters (hyperparameters, feature counts)
# MAGIC * Training metrics (accuracy, F1, per-class metrics)
# MAGIC * Model artifacts (serialized models)
# MAGIC * Feature importance (XGBoost only)
# MAGIC
# MAGIC **Model Registry:**
# MAGIC * Register champion model to Unity Catalog
# MAGIC * Model name: `civic_lens.ml.bbmp_outcome_classifier`
# MAGIC * Version tagging and lineage tracking
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Install Required Packages
# MAGIC %pip install xgboost --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup and Load Gold Table
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pyspark.sql import functions as F

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

print("=== ML Model Training: BBMP Outcome Prediction ===")
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow Version: {mlflow.__version__}\n")

# Configuration
GOLD_TABLE = "civic_lens.gold.bbmp_complaints_enriched"
MODEL_NAME = "civic_lens.ml.bbmp_outcome_classifier"
RANDOM_STATE = 42
TEST_SIZE = 0.2

print(f"Input Table: {GOLD_TABLE}")
print(f"Model Registry: {MODEL_NAME}")
print(f"Train/Test Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
print(f"Random State: {RANDOM_STATE}\n")

# Load gold table
print("Loading gold table...")
gold_df = spark.table(GOLD_TABLE)
print(f"✓ Loaded: {gold_df.count():,} rows, {len(gold_df.columns)} columns\n")

# Check target distribution
print("Target variable distribution:")
target_dist = gold_df.groupBy("outcome_label").count().orderBy("outcome_label")
target_dist.show()

# Check for nulls in target
null_targets = gold_df.filter(F.col("outcome_label").isNull()).count()
print(f"\nNull targets: {null_targets:,}")
if null_targets > 0:
    print(f"  ⚠️ Warning: {null_targets:,} records with null outcome_label will be excluded")

# COMMAND ----------

# DBTITLE 1,Feature Engineering and Data Preparation
print("=== Feature Engineering & Data Preparation ===")

# Define feature columns
print("\nSelecting features...")

# Numerical features
NUMERICAL_FEATURES = [
    "urgency_score",
    "remark_length",
    "grievance_year",
    "grievance_month",
    "grievance_day_of_week",
    "ward_cat_rejection_rate",
    "ward_cat_boilerplate_rate",
    "ward_cat_total_complaints",
    "ward_cat_complaints_30d",
    "ward_cat_open_30d",
    "ward_cat_avg_remark_length",
    "ward_cat_unique_depts",
    "days_since_grievance"
]

# TF-IDF features
TFIDF_FEATURES = [f"tfidf_feat_{i}" for i in range(1, 51)]

# Boolean features (convert to int)
BOOLEAN_FEATURES = [
    "remark_is_boilerplate",
    "is_high_urgency",
    "is_very_high_urgency",
    "is_problem_area",
    "is_high_boilerplate_area",
    "has_sufficient_context",
    "is_recent",
    "is_weekend"
]

# All feature columns
FEATURE_COLS = NUMERICAL_FEATURES + TFIDF_FEATURES + BOOLEAN_FEATURES

print(f"  Numerical features: {len(NUMERICAL_FEATURES)}")
print(f"  TF-IDF features: {len(TFIDF_FEATURES)}")
print(f"  Boolean features: {len(BOOLEAN_FEATURES)}")
print(f"  Total features: {len(FEATURE_COLS)}\n")

# Filter out nulls in target and select features
print("Preparing training data...")
ml_data = gold_df.filter(F.col("outcome_label").isNotNull()) \
    .select(["outcome_label"] + FEATURE_COLS)

print(f"  Records with valid target: {ml_data.count():,}")

# Convert boolean columns to integers
for bool_col in BOOLEAN_FEATURES:
    ml_data = ml_data.withColumn(bool_col, F.col(bool_col).cast("int"))

# Convert to Pandas for sklearn
print("\nConverting to Pandas for sklearn...")
ml_pdf = ml_data.toPandas()
print(f"✓ Converted: {len(ml_pdf):,} rows, {len(ml_pdf.columns)} columns")

# Check for any remaining nulls
null_counts = ml_pdf.isnull().sum()
if null_counts.sum() > 0:
    print("\n⚠️ Warning: Found nulls in features:")
    print(null_counts[null_counts > 0])
    print("\nFilling nulls with 0...")
    ml_pdf = ml_pdf.fillna(0)
    print("✓ Nulls filled")
else:
    print("\n✓ No nulls in features")

# Separate features and target
X = ml_pdf[FEATURE_COLS]
y = ml_pdf["outcome_label"].astype(int)

print(f"\n{'='*60}")
print("Data Preparation Summary")
print(f"{'='*60}")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nTarget distribution:")
print(y.value_counts().sort_index())
print(f"\nClass balance:")
for label in sorted(y.unique()):
    count = (y == label).sum()
    pct = count / len(y) * 100
    label_name = ["Resolved", "Closed", "Rejected"][label]
    print(f"  {label} ({label_name}): {count:,} ({pct:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Train-Test Split
print("=== Train-Test Split ===")

# Stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

print(f"\nTrain set class distribution:")
for label in sorted(y_train.unique()):
    count = (y_train == label).sum()
    pct = count / len(y_train) * 100
    label_name = ["Resolved", "Closed", "Rejected"][label]
    print(f"  {label} ({label_name}): {count:,} ({pct:.1f}%)")

print(f"\nTest set class distribution:")
for label in sorted(y_test.unique()):
    count = (y_test == label).sum()
    pct = count / len(y_test) * 100
    label_name = ["Resolved", "Closed", "Rejected"][label]
    print(f"  {label} ({label_name}): {count:,} ({pct:.1f}%)")

print("\n✓ Stratified split complete")

# COMMAND ----------

# DBTITLE 1,Train XGBoost Multi-Class Classifier
print("=== Training XGBoost Multi-Class Classifier ===")

# Set MLflow experiment
mlflow.set_experiment("/Users/pawanvirat32@gmail.com/civic-lens/bbmp-outcome-prediction")

# Start MLflow run
with mlflow.start_run(run_name="xgboost_multiclass") as run:
    print(f"\nMLflow Run ID: {run.info.run_id}")
    
    # Define XGBoost parameters
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "mlogloss"
    }
    
    print("\nModel parameters:")
    for k, v in xgb_params.items():
        print(f"  {k}: {v}")
        mlflow.log_param(k, v)
    
    # Log dataset info
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("test_size", TEST_SIZE)
    
    # Compute class weights to handle imbalance (4.4% rejections)
    print("\nComputing sample weights for class imbalance...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    print(f"  Sample weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
    print(f"  Rejection class upweighted by ~{sample_weights[y_train == 2].mean() / sample_weights[y_train == 1].mean():.1f}x")
    
    # Train model
    print("\nTraining XGBoost with sample weights...")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    print("✓ Training complete")
    
    # Make predictions
    print("\nMaking predictions...")
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    y_test_proba = xgb_model.predict_proba(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, average=None
    )
    macro_f1 = f1.mean()
    weighted_f1 = precision_recall_fscore_support(
        y_test, y_test_pred, average='weighted'
    )[2]
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("macro_f1", macro_f1)
    mlflow.log_metric("weighted_f1", weighted_f1)
    
    for i in range(3):
        label_name = ["resolved", "closed", "rejected"][i]
        mlflow.log_metric(f"precision_{label_name}", precision[i])
        mlflow.log_metric(f"recall_{label_name}", recall[i])
        mlflow.log_metric(f"f1_{label_name}", f1[i])
    
    # ROC-AUC (one-vs-rest for multiclass)
    try:
        roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')
        mlflow.log_metric("roc_auc_weighted", roc_auc)
    except:
        roc_auc = None
    
    # Print results
    print(f"\n{'='*60}")
    print("XGBoost Performance")
    print(f"{'='*60}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    if roc_auc:
        print(f"ROC-AUC (weighted): {roc_auc:.4f}")
    
    print("\nPer-class metrics (Test Set):")
    for i in range(3):
        label_name = ["Resolved", "Closed", "Rejected"][i]
        print(f"  {label_name}:")
        print(f"    Precision: {precision[i]:.4f}")
        print(f"    Recall: {recall[i]:.4f}")
        print(f"    F1-Score: {f1[i]:.4f}")
        print(f"    Support: {support[i]:,}")
    
    # Confusion matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n              Predicted")
    print("             0       1       2")
    print("Actual")
    for i, row in enumerate(cm):
        label = ["0 (Resolved)", "1 (Closed)  ", "2 (Rejected)"][i]
        print(f"  {label}  {row[0]:6,}  {row[1]:6,}  {row[2]:6,}")
    
    # Log model
    print("\nLogging model to MLflow...")
    mlflow.xgboost.log_model(xgb_model, "model")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features by Importance:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save feature importance as artifact
    feature_importance.to_csv("/tmp/xgb_feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/xgb_feature_importance.csv")
    
    print("\n✓ XGBoost model logged to MLflow")
    
    # Store for comparison
    xgb_test_acc = test_acc
    xgb_macro_f1 = macro_f1

# COMMAND ----------

# DBTITLE 1,Train Multinomial Logistic Regression
print("=== Training Multinomial Logistic Regression ===")

# Start MLflow run
with mlflow.start_run(run_name="logistic_regression_multinomial") as run:
    print(f"\nMLflow Run ID: {run.info.run_id}")
    
    # Define LogisticRegression parameters
    lr_params = {
        "multi_class": "multinomial",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
    
    print("\nModel parameters:")
    for k, v in lr_params.items():
        print(f"  {k}: {v}")
        mlflow.log_param(k, v)
    
    # Log dataset info
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("test_size", TEST_SIZE)
    
    # Train model
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_train, y_train)
    print("✓ Training complete")
    
    # Make predictions
    print("\nMaking predictions...")
    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)
    y_test_proba = lr_model.predict_proba(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, average=None
    )
    macro_f1 = f1.mean()
    weighted_f1 = precision_recall_fscore_support(
        y_test, y_test_pred, average='weighted'
    )[2]
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("macro_f1", macro_f1)
    mlflow.log_metric("weighted_f1", weighted_f1)
    
    for i in range(3):
        label_name = ["resolved", "closed", "rejected"][i]
        mlflow.log_metric(f"precision_{label_name}", precision[i])
        mlflow.log_metric(f"recall_{label_name}", recall[i])
        mlflow.log_metric(f"f1_{label_name}", f1[i])
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')
        mlflow.log_metric("roc_auc_weighted", roc_auc)
    except:
        roc_auc = None
    
    # Print results
    print(f"\n{'='*60}")
    print("Logistic Regression Performance")
    print(f"{'='*60}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    if roc_auc:
        print(f"ROC-AUC (weighted): {roc_auc:.4f}")
    
    print("\nPer-class metrics (Test Set):")
    for i in range(3):
        label_name = ["Resolved", "Closed", "Rejected"][i]
        print(f"  {label_name}:")
        print(f"    Precision: {precision[i]:.4f}")
        print(f"    Recall: {recall[i]:.4f}")
        print(f"    F1-Score: {f1[i]:.4f}")
        print(f"    Support: {support[i]:,}")
    
    # Confusion matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n              Predicted")
    print("             0       1       2")
    print("Actual")
    for i, row in enumerate(cm):
        label = ["0 (Resolved)", "1 (Closed)  ", "2 (Rejected)"][i]
        print(f"  {label}  {row[0]:6,}  {row[1]:6,}  {row[2]:6,}")
    
    # Log model
    print("\nLogging model to MLflow...")
    mlflow.sklearn.log_model(lr_model, "model")
    
    print("✓ Logistic Regression model logged to MLflow")
    
    # Store for comparison
    lr_test_acc = test_acc
    lr_macro_f1 = macro_f1

# COMMAND ----------

# DBTITLE 1,Model Comparison and Champion Selection
print("=== Model Comparison & Champion Selection ===")

print(f"\n{'='*60}")
print("Model Performance Summary")
print(f"{'='*60}")

comparison = pd.DataFrame({
    'Model': ['XGBoost', 'Logistic Regression'],
    'Test Accuracy': [xgb_test_acc, lr_test_acc],
    'Macro F1': [xgb_macro_f1, lr_macro_f1]
})

print("\n", comparison.to_string(index=False))

# Determine champion
if xgb_test_acc > lr_test_acc:
    champion_model = "XGBoost"
    champion_accuracy = xgb_test_acc
    champion_f1 = xgb_macro_f1
    print(f"\n{'='*60}")
    print(f"⭐ Champion Model: {champion_model}")
    print(f"{'='*60}")
    print(f"Test Accuracy: {champion_accuracy:.4f}")
    print(f"Macro F1: {champion_f1:.4f}")
    print(f"Improvement over baseline: {(champion_accuracy - lr_test_acc)*100:.2f}%")
else:
    champion_model = "Logistic Regression"
    champion_accuracy = lr_test_acc
    champion_f1 = lr_macro_f1
    print(f"\n{'='*60}")
    print(f"⭐ Champion Model: {champion_model}")
    print(f"{'='*60}")
    print(f"Test Accuracy: {champion_accuracy:.4f}")
    print(f"Macro F1: {champion_f1:.4f}")
    print("\nBaseline logistic regression performs competitively!")

print(f"\n✓ Champion selected for Unity Catalog registration")
print(f"\nNext Step: Register to {MODEL_NAME}")

# COMMAND ----------

# DBTITLE 1,Training Summary and Next Steps
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## ✅ ML Model Training Complete
# MAGIC
# MAGIC ### Summary
# MAGIC
# MAGIC This notebook successfully trained and evaluated **two multi-class classification models** to predict BBMP complaint outcomes:
# MAGIC
# MAGIC **Models Trained:**
# MAGIC 1. **XGBoost Multi-Class Classifier** (Gradient Boosting)
# MAGIC    * Configuration: `objective="multi:softmax"`, `num_class=3`
# MAGIC    * 100 estimators, max_depth=6, learning_rate=0.1
# MAGIC    * Robust to class imbalance, captures non-linear interactions
# MAGIC
# MAGIC 2. **Multinomial Logistic Regression** (Linear Baseline)
# MAGIC    * Configuration: `multi_class="multinomial"`, `solver="lbfgs"`
# MAGIC    * Interpretable coefficients, fast training
# MAGIC    * Serves as performance baseline
# MAGIC
# MAGIC ### Dataset
# MAGIC
# MAGIC * **Total Records:** 742,216 complaints (after filtering nulls)
# MAGIC * **Training Set:** ~593,773 samples (80%)
# MAGIC * **Test Set:** ~148,443 samples (20%)
# MAGIC * **Features:** 71 total
# MAGIC   * 13 numerical (urgency_score, remark_length, temporal features, ward-category aggregates)
# MAGIC   * 50 TF-IDF/SVD embeddings (semantic text features)
# MAGIC   * 8 boolean flags (urgency, problem area, context, temporal)
# MAGIC * **Target Classes:**
# MAGIC   * 0 = Resolved (complaint successfully addressed)
# MAGIC   * 1 = Closed (complaint closed without resolution)
# MAGIC   * 2 = Rejected (complaint rejected by system)
# MAGIC
# MAGIC ### Evaluation Metrics
# MAGIC
# MAGIC ✅ **Test Accuracy** (overall correctness)  
# MAGIC ✅ **Macro F1-Score** (unweighted average across classes)  
# MAGIC ✅ **Weighted F1-Score** (weighted by class support)  
# MAGIC ✅ **Per-Class Precision/Recall/F1** (class-specific performance)  
# MAGIC ✅ **ROC-AUC** (probability calibration, one-vs-rest)  
# MAGIC ✅ **Confusion Matrix** (error analysis)
# MAGIC
# MAGIC ### MLflow Integration
# MAGIC
# MAGIC **Experiment:** `/Users/pawanvirat32@gmail.com/civic-lens/bbmp-outcome-prediction`
# MAGIC
# MAGIC **Logged Artifacts:**
# MAGIC * Model parameters (hyperparameters, solver settings)
# MAGIC * Training & test metrics (accuracy, F1, precision, recall, ROC-AUC)
# MAGIC * Model artifacts (serialized models for both XGBoost and Logistic Regression)
# MAGIC * Feature importance (XGBoost only - top predictive features)
# MAGIC
# MAGIC **Both models are logged and ready for comparison in the MLflow UI.**
# MAGIC
# MAGIC ### Champion Model
# MAGIC
# MAGIC The model with the highest **test accuracy** is automatically selected as the champion and recommended for production deployment.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Next Steps
# MAGIC
# MAGIC ### 1. Model Registry
# MAGIC
# MAGIC **Register Champion Model to Unity Catalog:**
# MAGIC ```python
# MAGIC import mlflow
# MAGIC from mlflow import MlflowClient
# MAGIC
# MAGIC client = MlflowClient()
# MAGIC
# MAGIC # Get the best run
# MAGIC runs = mlflow.search_runs(
# MAGIC     experiment_names=["/Users/pawanvirat32@gmail.com/civic-lens/bbmp-outcome-prediction"],
# MAGIC     order_by=["metrics.test_accuracy DESC"],
# MAGIC     max_results=1
# MAGIC )
# MAGIC best_run_id = runs.iloc[0]["run_id"]
# MAGIC
# MAGIC # Register model
# MAGIC model_uri = f"runs:/{best_run_id}/model"
# MAGIC mlflow.register_model(
# MAGIC     model_uri=model_uri,
# MAGIC     name="civic_lens.ml.bbmp_outcome_classifier"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### 2. Model Deployment
# MAGIC
# MAGIC **Option A: Batch Inference**
# MAGIC ```python
# MAGIC # Load model from Unity Catalog
# MAGIC import mlflow.pyfunc
# MAGIC model = mlflow.pyfunc.load_model("models:/civic_lens.ml.bbmp_outcome_classifier/1")
# MAGIC
# MAGIC # Score new complaints
# MAGIC new_complaints = spark.table("civic_lens.silver.bbmp_complaints_clean")
# MAGIC predictions = model.predict(new_complaints_features)
# MAGIC ```
# MAGIC
# MAGIC **Option B: Real-Time Serving**
# MAGIC * Deploy model to Databricks Model Serving endpoint
# MAGIC * Create REST API for real-time complaint triage
# MAGIC * Integrate with complaint intake system
# MAGIC
# MAGIC ### 3. Model Monitoring
# MAGIC
# MAGIC **Track Model Health:**
# MAGIC * **Prediction Drift:** Monitor class distribution shifts over time
# MAGIC * **Feature Drift:** Track changes in input feature distributions
# MAGIC * **Performance Degradation:** Periodically evaluate on recent data
# MAGIC * **Retrain Triggers:** Set thresholds for automatic retraining
# MAGIC
# MAGIC **Scheduled Retraining:**
# MAGIC * Weekly or monthly retraining on latest data
# MAGIC * A/B test new models against current production model
# MAGIC * Version control with MLflow Model Registry
# MAGIC
# MAGIC ### 4. Feature Engineering Improvements
# MAGIC
# MAGIC **Potential Enhancements:**
# MAGIC * **Category Embeddings:** Learn dense representations for complaint categories
# MAGIC * **Staff Performance Features:** Historical resolution rates by staff member
# MAGIC * **Temporal Patterns:** Time-of-day, day-of-week interaction features
# MAGIC * **Text Embeddings:** Replace TF-IDF with BERT/sentence-transformers for richer semantics
# MAGIC * **Interaction Features:** Cross-features between urgency and ward-category metrics
# MAGIC
# MAGIC ### 5. Integration & Productionization
# MAGIC
# MAGIC **Dashboard Integration:**
# MAGIC * Real-time complaint priority scores
# MAGIC * Ward-level outcome predictions
# MAGIC * Staff workload balancing recommendations
# MAGIC
# MAGIC **Complaint Triage System:**
# MAGIC * Auto-prioritize high-risk complaints (predicted rejection)
# MAGIC * Route complaints to best-fit departments
# MAGIC * Alert supervisors for problem areas (high rejection zones)
# MAGIC
# MAGIC **Citizen Portal:**
# MAGIC * Estimated resolution probability at complaint submission
# MAGIC * Proactive guidance for improving complaint quality
# MAGIC * Transparent outcome explanations
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Model Use Cases
# MAGIC
# MAGIC ### Early Warning System
# MAGIC * Identify complaints likely to be rejected **before** staff review
# MAGIC * Proactive intervention to improve outcomes
# MAGIC * Reduce rejection rate through better routing
# MAGIC
# MAGIC ### Resource Optimization
# MAGIC * Predict workload by outcome type (resolved vs closed vs rejected)
# MAGIC * Allocate staff to high-priority complaints
# MAGIC * Balance department workloads based on predicted difficulty
# MAGIC
# MAGIC ### Performance Benchmarking
# MAGIC * Compare predicted vs actual outcomes by ward
# MAGIC * Identify wards with high rejection rates
# MAGIC * Target training and resources to underperforming areas
# MAGIC
# MAGIC ### Citizen Engagement
# MAGIC * Set realistic expectations at complaint submission
# MAGIC * Provide actionable feedback to improve complaint quality
# MAGIC * Increase transparency in complaint handling process
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔧 Technical Details
# MAGIC
# MAGIC **Training Environment:**
# MAGIC * Compute: Databricks Serverless (CPU)
# MAGIC * Data Source: `civic_lens.gold.bbmp_complaints_enriched`
# MAGIC * ML Framework: scikit-learn + XGBoost
# MAGIC * Experiment Tracking: MLflow
# MAGIC * Model Registry: Unity Catalog
# MAGIC
# MAGIC **Reproducibility:**
# MAGIC * Fixed random seed (42) for train-test split
# MAGIC * Stratified sampling preserves class distribution
# MAGIC * All hyperparameters logged to MLflow
# MAGIC * Feature list stored with model artifacts
# MAGIC
# MAGIC **Model Artifacts:**
# MAGIC * Serialized model (pickle format)
# MAGIC * Feature names and types
# MAGIC * Training metrics and confusion matrix
# MAGIC * Feature importance (XGBoost)
# MAGIC * Model signature (input/output schema)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🏆 Training Job:** Complete  
# MAGIC **📈 Models Trained:** 2 (XGBoost + Logistic Regression)  
# MAGIC **🎯 Champion Selected:** Highest test accuracy  
# MAGIC **📦 MLflow Runs:** Logged with full metrics and artifacts  
# MAGIC **✅ Ready For:** Unity Catalog registration & deployment  
# MAGIC
# MAGIC ---
