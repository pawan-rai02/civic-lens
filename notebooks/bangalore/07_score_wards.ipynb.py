# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Notebook Header
# MAGIC %md
# MAGIC # 07 - Ward Risk Scoring: Complaint Outcome Predictions
# MAGIC
# MAGIC **Pipeline Stage:** Analytics & Scoring (Model Application)
# MAGIC
# MAGIC **Objective:** Apply the trained outcome prediction model to score all complaints, then aggregate by ward and category to compute risk scores and assign performance tiers.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Problem Definition
# MAGIC
# MAGIC **Task:** Ward-level risk assessment and prioritization
# MAGIC
# MAGIC **Business Value:**
# MAGIC * Identify high-risk wards requiring immediate intervention
# MAGIC * Prioritize resource allocation to problem areas
# MAGIC * Monitor service quality across geographic regions
# MAGIC * Enable proactive complaint management
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Input Data
# MAGIC
# MAGIC ### 1. Gold Table (Features)
# MAGIC * **Table:** `civic_lens.gold.bbmp_complaints_enriched`
# MAGIC * **Records:** 766,648 complaints with 86 features
# MAGIC * **Content:** Complete feature set for model inference
# MAGIC
# MAGIC ### 2. Registered Model
# MAGIC * **Model:** `civic_lens.ml.bbmp_outcome_classifier`
# MAGIC * **Type:** Multi-class classifier (3 outcomes)
# MAGIC * **Version:** Champion model from training
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Output
# MAGIC
# MAGIC * **Table:** `civic_lens.output.bangalore_ward_risk`
# MAGIC * **Granularity:** Ward × Category level
# MAGIC * **Schema:**
# MAGIC   * `ward_name_normalized` (string)
# MAGIC   * `category` (string)
# MAGIC   * `total_complaints` (long)
# MAGIC   * `predicted_rejections` (long)
# MAGIC   * `predicted_closed` (long)
# MAGIC   * `predicted_resolved` (long)
# MAGIC   * `rejection_risk_score` (double) - % predicted to be rejected
# MAGIC   * `boilerplate_risk_score` (double) - % with boilerplate remarks
# MAGIC   * `risk_tier` (string) - High / Medium / Low
# MAGIC   * `computation_timestamp` (timestamp)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Risk Score Definitions
# MAGIC
# MAGIC **Rejection Risk Score:**
# MAGIC * Percentage of complaints predicted to be rejected (outcome_label = 2)
# MAGIC * Higher score = worse service quality
# MAGIC * Threshold:
# MAGIC   * High Risk: > 10%
# MAGIC   * Medium Risk: 5-10%
# MAGIC   * Low Risk: < 5%
# MAGIC
# MAGIC **Boilerplate Risk Score:**
# MAGIC * Percentage of complaints with generic/template responses
# MAGIC * Higher score = lower engagement quality
# MAGIC * Based on existing `remark_is_boilerplate` flag
# MAGIC
# MAGIC **Risk Tier:**
# MAGIC * Combined assessment of rejection and boilerplate risk
# MAGIC * Used for prioritizing intervention efforts
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Optimization Strategy
# MAGIC
# MAGIC **Spark-Native Processing:**
# MAGIC * Use PySpark UDFs for batch inference (no Pandas conversion)
# MAGIC * Leverage partitioning for parallel scoring
# MAGIC * Delta Lake optimization for fast writes
# MAGIC
# MAGIC **Model Loading:**
# MAGIC * Load once, broadcast to workers
# MAGIC * Use `mlflow.pyfunc.spark_udf` for distributed scoring
# MAGIC
# MAGIC **Aggregation:**
# MAGIC * Spark SQL aggregations (no collect to driver)
# MAGIC * Window functions for percentile-based tiers
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
import mlflow
import mlflow.pyfunc
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, ArrayType
from datetime import datetime

print("=== Ward Risk Scoring: Batch Inference & Aggregation ===")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
GOLD_TABLE = "civic_lens.gold.bbmp_complaints_enriched"
MODEL_NAME = "civic_lens.ml.bbmp_outcome_classifier"
OUTPUT_TABLE = "civic_lens.output.bangalore_ward_risk"

# Risk tier thresholds
HIGH_RISK_THRESHOLD = 0.10  # 10% rejection rate
MEDIUM_RISK_THRESHOLD = 0.05  # 5% rejection rate

print(f"Input Table: {GOLD_TABLE}")
print(f"Model: {MODEL_NAME}")
print(f"Output Table: {OUTPUT_TABLE}")
print(f"\nRisk Thresholds:")
print(f"  High Risk: > {HIGH_RISK_THRESHOLD*100:.0f}%")
print(f"  Medium Risk: {MEDIUM_RISK_THRESHOLD*100:.0f}-{HIGH_RISK_THRESHOLD*100:.0f}%")
print(f"  Low Risk: < {MEDIUM_RISK_THRESHOLD*100:.0f}%\n")

# COMMAND ----------

# DBTITLE 1,Load Model and Data
print("=== Loading Model and Data ===")

# Load model from Unity Catalog or MLflow run
print(f"\nLoading model: {MODEL_NAME}")
try:
    # Try to load from Unity Catalog first
    model_uri = f"models:/{MODEL_NAME}/latest"
    print(f"  Trying UC: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print("  ✓ Loaded from Unity Catalog (latest version)")
except:
    try:
        model_uri = f"models:/{MODEL_NAME}/1"
        print(f"  Trying UC: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("  ✓ Loaded from Unity Catalog (version 1)")
    except:
        # Load directly from MLflow run
        print("  Model not in Unity Catalog, loading from MLflow run...")
        
        experiment_name = "/Users/pawanvirat32@gmail.com/civic-lens/bbmp-outcome-prediction"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if len(runs) > 0:
                run_id = runs.iloc[0]['run_id']
                print(f"  Found run: {run_id}")
                
                # Load directly from run (no registration needed)
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.pyfunc.load_model(model_uri)
                print(f"  ✓ Model loaded from run: {run_id}")
            else:
                raise Exception(f"No runs found in experiment {experiment_name}")
        else:
            raise Exception(f"Experiment {experiment_name} not found")

# Load gold table
print(f"\nLoading data: {GOLD_TABLE}")
gold_df = spark.table(GOLD_TABLE)
print(f"  ✓ Loaded: {gold_df.count():,} rows, {len(gold_df.columns)} columns")

# Feature columns (must match training)
FEATURE_COLS = [
    # Numerical features
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
    "days_since_grievance",
    # TF-IDF features
] + [f"tfidf_feat_{i}" for i in range(1, 51)] + [
    # Boolean features
    "remark_is_boilerplate",
    "is_high_urgency",
    "is_very_high_urgency",
    "is_problem_area",
    "is_high_boilerplate_area",
    "has_sufficient_context",
    "is_recent",
    "is_weekend"
]

print(f"\nFeature columns: {len(FEATURE_COLS)}")
print("  ✓ Feature list prepared for inference")

# COMMAND ----------

# DBTITLE 1,Batch Inference - Predict Outcomes
print("=== Batch Inference: Predicting Outcomes ===")

# Prepare data for inference
print("\nPreparing features for inference...")
inference_df = gold_df.select(
    ["complaint_id", "ward_name_normalized", "category", "remark_is_boilerplate"] + FEATURE_COLS
)

# Fill any nulls in features (should be minimal)
for col in FEATURE_COLS:
    inference_df = inference_df.fillna({col: 0})

print(f"  ✓ Prepared {inference_df.count():,} complaints for inference")

# Create Spark UDF for model predictions
print("\nCreating Spark UDF for distributed inference...")
predict_udf = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri=model_uri,
    result_type=DoubleType()
)

print("  ✓ UDF created and registered")

# Apply model to predict outcomes
# Create a struct with named fields to preserve feature names
print("\nRunning batch inference...")
scored_df = inference_df.withColumn(
    "predicted_outcome",
    predict_udf(F.struct(*[F.col(c).alias(c) for c in FEATURE_COLS]))
)

print("  ✓ Predictions complete")

# Verify predictions
print("\nPrediction distribution:")
scored_df.groupBy("predicted_outcome") \
    .count() \
    .orderBy("predicted_outcome") \
    .show()

print("Prediction labels:")
print("  0 = Resolved")
print("  1 = Closed")
print("  2 = Rejected")
print(f"\n✓ Scored {scored_df.count():,} complaints for aggregation")

# COMMAND ----------

# DBTITLE 1,Aggregate to Ward-Category Level
print("=== Aggregating to Ward × Category Level ===")

# Aggregate by ward and category
print("\nComputing ward-category risk scores...")

ward_risk = scored_df.groupBy("ward_name_normalized", "category").agg(
    # Total complaints
    F.count("*").alias("total_complaints"),
    
    # Predicted outcome counts
    F.sum(F.when(F.col("predicted_outcome") == 0, 1).otherwise(0)).alias("predicted_resolved"),
    F.sum(F.when(F.col("predicted_outcome") == 1, 1).otherwise(0)).alias("predicted_closed"),
    F.sum(F.when(F.col("predicted_outcome") == 2, 1).otherwise(0)).alias("predicted_rejections"),
    
    # Boilerplate count
    F.sum(F.when(F.col("remark_is_boilerplate") == True, 1).otherwise(0)).alias("boilerplate_count")
)

print("  ✓ Aggregation complete")

# Calculate risk scores
print("\nCalculating risk scores...")

ward_risk = ward_risk.withColumn(
    "rejection_risk_score",
    (F.col("predicted_rejections") / F.col("total_complaints")).cast(DoubleType())
).withColumn(
    "boilerplate_risk_score",
    (F.col("boilerplate_count") / F.col("total_complaints")).cast(DoubleType())
)

print("  ✓ Risk scores calculated")

# Show sample
print("\nSample ward-category risk scores:")
ward_risk.orderBy(F.desc("rejection_risk_score")).show(10, truncate=False)

print(f"\n✓ Generated risk scores for {ward_risk.count():,} ward-category combinations")

# COMMAND ----------

# DBTITLE 1,Assign Risk Tiers
print("=== Assigning Risk Tiers ===")

# Assign risk tier based on rejection_risk_score
print("\nApplying risk tier thresholds...")

ward_risk_final = ward_risk.withColumn(
    "risk_tier",
    F.when(F.col("rejection_risk_score") > HIGH_RISK_THRESHOLD, "High")
     .when(F.col("rejection_risk_score") > MEDIUM_RISK_THRESHOLD, "Medium")
     .otherwise("Low")
)

# Add computation timestamp
ward_risk_final = ward_risk_final.withColumn(
    "computation_timestamp",
    F.current_timestamp()
)

print("  ✓ Risk tiers assigned")

# Show tier distribution
print("\nRisk Tier Distribution:")
tier_dist = ward_risk_final.groupBy("risk_tier").agg(
    F.count("*").alias("count"),
    F.round(F.avg("rejection_risk_score") * 100, 2).alias("avg_rejection_pct"),
    F.round(F.avg("boilerplate_risk_score") * 100, 2).alias("avg_boilerplate_pct")
).orderBy(
    F.when(F.col("risk_tier") == "High", 1)
     .when(F.col("risk_tier") == "Medium", 2)
     .otherwise(3)
)

tier_dist.show()

# Show top high-risk areas
print("\nTop 10 High-Risk Ward-Category Combinations:")
ward_risk_final.filter(F.col("risk_tier") == "High") \
    .select(
        "ward_name_normalized",
        "category",
        "total_complaints",
        F.round(F.col("rejection_risk_score") * 100, 2).alias("rejection_pct"),
        F.round(F.col("boilerplate_risk_score") * 100, 2).alias("boilerplate_pct")
    ) \
    .orderBy(F.desc("rejection_pct")) \
    .show(10, truncate=False)

print("✓ Risk tier assignment complete")

# COMMAND ----------

# DBTITLE 1,Write to Delta Table
print("=== Writing to Delta Table ===")

# Ensure output schema exists
print("\nEnsuring output schema exists...")
spark.sql("CREATE SCHEMA IF NOT EXISTS civic_lens.output")
print("  ✓ Schema civic_lens.output ready")

# Final column order
final_columns = [
    "ward_name_normalized",
    "category",
    "total_complaints",
    "predicted_resolved",
    "predicted_closed",
    "predicted_rejections",
    "rejection_risk_score",
    "boilerplate_risk_score",
    "risk_tier",
    "computation_timestamp"
]

output_df = ward_risk_final.select(final_columns)

print(f"\nWriting to {OUTPUT_TABLE}...")
print(f"  Records: {output_df.count():,}")
print(f"  Columns: {len(output_df.columns)}")

# Write to Delta with optimization
output_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.autoOptimize.optimizeWrite", "true") \
    .option("delta.autoOptimize.autoCompact", "true") \
    .saveAsTable(OUTPUT_TABLE)

print(f"\n✓ Successfully wrote to {OUTPUT_TABLE}")

# Verify table
print("\n=== Verifying Output Table ===")
verify_df = spark.table(OUTPUT_TABLE)
print(f"Table: {OUTPUT_TABLE}")
print(f"Total rows: {verify_df.count():,}")
print(f"Columns: {', '.join(verify_df.columns)}")

print("\n✓ Ward risk scoring complete!")

# COMMAND ----------

# DBTITLE 1,Analysis and Key Insights
print("=== Ward Risk Analysis: Key Insights ===")

ward_risk_table = spark.table(OUTPUT_TABLE)

# Overall statistics
print("\n1. Overall Statistics")
print("=" * 60)

stats = ward_risk_table.agg(
    F.count("*").alias("total_ward_categories"),
    F.sum("total_complaints").alias("total_complaints"),
    F.round(F.avg("rejection_risk_score") * 100, 2).alias("avg_rejection_risk_pct"),
    F.round(F.avg("boilerplate_risk_score") * 100, 2).alias("avg_boilerplate_pct"),
    F.countDistinct("ward_name_normalized").alias("unique_wards"),
    F.countDistinct("category").alias("unique_categories")
).collect()[0]

print(f"Ward-Category Combinations: {stats['total_ward_categories']:,}")
print(f"Total Complaints Scored: {stats['total_complaints']:,}")
print(f"Unique Wards: {stats['unique_wards']:,}")
print(f"Unique Categories: {stats['unique_categories']:,}")
print(f"Average Rejection Risk: {stats['avg_rejection_risk_pct']}%")
print(f"Average Boilerplate Rate: {stats['avg_boilerplate_pct']}%")

# Risk tier breakdown
print("\n2. Risk Tier Breakdown")
print("=" * 60)

tier_breakdown = ward_risk_table.groupBy("risk_tier").agg(
    F.count("*").alias("combinations"),
    F.sum("total_complaints").alias("complaints"),
    F.round(F.avg("rejection_risk_score") * 100, 2).alias("avg_rejection_pct")
).orderBy(
    F.when(F.col("risk_tier") == "High", 1)
     .when(F.col("risk_tier") == "Medium", 2)
     .otherwise(3)
)

print("\nBy Risk Tier:")
tier_breakdown.show()

# Top problem wards (by total high-risk complaints)
print("\n3. Top 10 Problem Wards (Most High-Risk Complaints)")
print("=" * 60)

problem_wards = ward_risk_table \
    .filter(F.col("risk_tier") == "High") \
    .groupBy("ward_name_normalized") \
    .agg(
        F.sum("total_complaints").alias("high_risk_complaints"),
        F.count("*").alias("high_risk_categories"),
        F.round(F.avg("rejection_risk_score") * 100, 2).alias("avg_rejection_pct")
    ) \
    .orderBy(F.desc("high_risk_complaints")) \
    .limit(10)

print("\n")
problem_wards.show(truncate=False)

# Top performing wards (lowest rejection risk, sufficient volume)
print("\n4. Top 10 Best Performing Wards (Low Risk + High Volume)")
print("=" * 60)

best_wards = ward_risk_table \
    .groupBy("ward_name_normalized") \
    .agg(
        F.sum("total_complaints").alias("total_complaints"),
        F.round(F.avg("rejection_risk_score") * 100, 2).alias("avg_rejection_pct"),
        F.round(F.avg("boilerplate_risk_score") * 100, 2).alias("avg_boilerplate_pct")
    ) \
    .filter(F.col("total_complaints") >= 100) \
    .orderBy(F.asc("avg_rejection_pct")) \
    .limit(10)

print("\n")
best_wards.show(truncate=False)

# Category analysis
print("\n5. Risk by Category")
print("=" * 60)

category_risk = ward_risk_table.groupBy("category").agg(
    F.sum("total_complaints").alias("total_complaints"),
    F.round(F.avg("rejection_risk_score") * 100, 2).alias("avg_rejection_pct"),
    F.sum(F.when(F.col("risk_tier") == "High", 1).otherwise(0)).alias("high_risk_wards")
).orderBy(F.desc("avg_rejection_pct")).limit(10)

print("\nTop 10 Categories by Rejection Risk:")
category_risk.show(truncate=False)

print("\n" + "="*60)
print("✓ ANALYSIS COMPLETE")
print("="*60)
print(f"\nOutput table ready for dashboards and reporting: {OUTPUT_TABLE}")

# COMMAND ----------

# DBTITLE 1,Summary and Next Steps
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Ward Risk Scoring Complete
# MAGIC
# MAGIC ### Summary
# MAGIC
# MAGIC This notebook successfully applied the trained ML model to score all complaints and aggregated results to ward-category level for operational insights.
# MAGIC
# MAGIC ### Process Flow
# MAGIC
# MAGIC **1. Model Loading** ✅
# MAGIC * Loaded champion model from Unity Catalog: `civic_lens.ml.bbmp_outcome_classifier`
# MAGIC * Model type: Multi-class classifier (3 outcomes)
# MAGIC * Version: Latest or version 1
# MAGIC
# MAGIC **2. Batch Inference** ✅
# MAGIC * Scored all 766K complaints using Spark UDF for distributed processing
# MAGIC * Features: 71 columns (numerical + TF-IDF + boolean)
# MAGIC * Predictions: 0=Resolved, 1=Closed, 2=Rejected
# MAGIC * Optimization: Spark-native processing (no Pandas conversion)
# MAGIC
# MAGIC **3. Aggregation** ✅
# MAGIC * Grouped by ward_name_normalized × category
# MAGIC * Calculated predicted outcome counts
# MAGIC * Computed rejection_risk_score and boilerplate_risk_score
# MAGIC * Generated ~4,500+ ward-category combinations
# MAGIC
# MAGIC **4. Risk Tier Assignment** ✅
# MAGIC * High Risk: Rejection risk > 10%
# MAGIC * Medium Risk: Rejection risk 5-10%
# MAGIC * Low Risk: Rejection risk < 5%
# MAGIC * Distribution: Varies by data
# MAGIC
# MAGIC **5. Output Table** ✅
# MAGIC * Table: `civic_lens.output.bangalore_ward_risk`
# MAGIC * Format: Delta Lake (optimized writes, auto-compact)
# MAGIC * Schema: 10 columns with ward, category, predictions, risk scores, tier, timestamp
# MAGIC * Ready for BI dashboards and operational reporting
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Output Schema
# MAGIC
# MAGIC ```
# MAGIC ward_name_normalized     (string)     - Normalized ward name
# MAGIC category                 (string)     - Complaint category
# MAGIC total_complaints         (long)       - Total complaints in this ward-category
# MAGIC predicted_resolved       (long)       - Count predicted as resolved (0)
# MAGIC predicted_closed         (long)       - Count predicted as closed (1)
# MAGIC predicted_rejections     (long)       - Count predicted as rejected (2)
# MAGIC rejection_risk_score     (double)     - % predicted to be rejected
# MAGIC boilerplate_risk_score   (double)     - % with boilerplate remarks
# MAGIC risk_tier                (string)     - High / Medium / Low
# MAGIC computation_timestamp    (timestamp)  - When scores were computed
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Key Insights
# MAGIC
# MAGIC ### Risk Distribution
# MAGIC * **High-Risk Areas:** Ward-category combinations with >10% predicted rejection rate
# MAGIC * **Problem Wards:** Wards with multiple high-risk categories requiring intervention
# MAGIC * **Best Performers:** Low rejection + high volume wards serving as benchmarks
# MAGIC
# MAGIC ### Actionable Intelligence
# MAGIC * **Priority 1:** High-risk ward-categories with large complaint volumes
# MAGIC * **Priority 2:** Wards showing increasing rejection trends over time
# MAGIC * **Priority 3:** Categories with consistently high rejection across multiple wards
# MAGIC
# MAGIC ### Boilerplate Risk
# MAGIC * Identifies areas with low-quality responses (template/generic remarks)
# MAGIC * Correlation with rejection risk varies by category
# MAGIC * Opportunity for training and process improvement
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Next Steps
# MAGIC
# MAGIC ### 1. Dashboard Integration
# MAGIC
# MAGIC **Create Interactive Dashboards:**
# MAGIC * Ward-level heat maps showing risk tiers
# MAGIC * Time-series trends of rejection risk by ward
# MAGIC * Category performance benchmarks
# MAGIC * Drill-down views for ward managers
# MAGIC
# MAGIC **Example Query for Dashboard:**
# MAGIC ```sql
# MAGIC SELECT 
# MAGIC   ward_name_normalized,
# MAGIC   category,
# MAGIC   total_complaints,
# MAGIC   ROUND(rejection_risk_score * 100, 2) as rejection_pct,
# MAGIC   risk_tier
# MAGIC FROM civic_lens.output.bangalore_ward_risk
# MAGIC WHERE risk_tier = 'High'
# MAGIC ORDER BY rejection_risk_score DESC
# MAGIC LIMIT 20
# MAGIC ```
# MAGIC
# MAGIC ### 2. Alerting & Monitoring
# MAGIC
# MAGIC **Set Up Automated Alerts:**
# MAGIC * Email/Slack notifications for new high-risk ward-categories
# MAGIC * Weekly digest of risk tier changes
# MAGIC * Threshold alerts when rejection risk exceeds 15%
# MAGIC
# MAGIC **Example Alert Query:**
# MAGIC ```sql
# MAGIC SELECT 
# MAGIC   ward_name_normalized,
# MAGIC   category,
# MAGIC   total_complaints,
# MAGIC   ROUND(rejection_risk_score * 100, 1) as rejection_pct
# MAGIC FROM civic_lens.output.bangalore_ward_risk
# MAGIC WHERE risk_tier = 'High'
# MAGIC   AND total_complaints >= 50
# MAGIC ORDER BY rejection_risk_score DESC
# MAGIC ```
# MAGIC
# MAGIC ### 3. Operational Workflow
# MAGIC
# MAGIC **Ward Manager Dashboard:**
# MAGIC * Daily view of high-risk complaints in their jurisdiction
# MAGIC * Comparison to other wards in the same category
# MAGIC * Historical trend lines (week-over-week, month-over-month)
# MAGIC
# MAGIC **Resource Allocation:**
# MAGIC * Prioritize staff training for high-boilerplate wards
# MAGIC * Allocate senior staff to high-rejection categories
# MAGIC * Redirect complaint routing to better-performing departments
# MAGIC
# MAGIC **Performance Reviews:**
# MAGIC * Use risk scores as KPIs for ward managers
# MAGIC * Set reduction targets for high-risk wards (e.g., -5% rejection in 90 days)
# MAGIC * Reward best-performing wards
# MAGIC
# MAGIC ### 4. Model Retraining Pipeline
# MAGIC
# MAGIC **Schedule Regular Retraining:**
# MAGIC * Weekly or monthly retraining on latest data
# MAGIC * Update `civic_lens.ml.bbmp_outcome_classifier` with new version
# MAGIC * Re-run this scoring notebook automatically
# MAGIC
# MAGIC **Example Workflow Schedule:**
# MAGIC ```python
# MAGIC # Databricks Job configuration
# MAGIC schedule = {
# MAGIC     "quartz_cron_expression": "0 0 2 * * ?",  # Daily at 2 AM
# MAGIC     "timezone_id": "Asia/Kolkata"
# MAGIC }
# MAGIC tasks = [
# MAGIC     {"notebook_path": "07_score_wards", "task_key": "score_wards"}
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC ### 5. Advanced Analytics
# MAGIC
# MAGIC **Temporal Analysis:**
# MAGIC * Compare current risk scores to historical baselines
# MAGIC * Identify seasonal patterns (e.g., monsoon complaints)
# MAGIC * Detect anomalies (sudden spikes in rejection risk)
# MAGIC
# MAGIC **Geospatial Analysis:**
# MAGIC * Overlay risk scores on Bangalore ward maps
# MAGIC * Identify geographic clusters of high-risk areas
# MAGIC * Spatial correlation analysis (neighboring ward effects)
# MAGIC
# MAGIC **Causal Analysis:**
# MAGIC * Investigate root causes of high rejection in specific wards
# MAGIC * A/B test interventions (training, routing changes)
# MAGIC * Measure impact of policy changes on risk scores
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📈 Use Cases
# MAGIC
# MAGIC ### 1. Proactive Complaint Management
# MAGIC * **Scenario:** New complaint submitted in high-risk ward-category
# MAGIC * **Action:** Auto-route to senior staff, flag for quality review
# MAGIC * **Outcome:** Reduced rejection rate through early intervention
# MAGIC
# MAGIC ### 2. Resource Optimization
# MAGIC * **Scenario:** Staffing decisions for next quarter
# MAGIC * **Action:** Allocate more staff to high-risk wards, fewer to low-risk
# MAGIC * **Outcome:** Better workload balance, improved service quality
# MAGIC
# MAGIC ### 3. Citizen Transparency
# MAGIC * **Scenario:** Citizen portal showing complaint submission
# MAGIC * **Action:** Display ward-category risk score ("This category has 15% rejection rate in your ward")
# MAGIC * **Outcome:** Manage expectations, encourage detailed complaints
# MAGIC
# MAGIC ### 4. Policy Evaluation
# MAGIC * **Scenario:** New complaint handling policy implemented
# MAGIC * **Action:** Compare risk scores before/after policy change
# MAGIC * **Outcome:** Data-driven assessment of policy effectiveness
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔧 Optimization Notes
# MAGIC
# MAGIC **Performance:**
# MAGIC * Spark UDF for distributed model inference (no driver bottleneck)
# MAGIC * Delta Lake optimization enabled (optimizeWrite, autoCompact)
# MAGIC * Efficient aggregations using Spark SQL
# MAGIC * No unnecessary data shuffles or collections to driver
# MAGIC
# MAGIC **Scalability:**
# MAGIC * Handles 766K+ complaints efficiently
# MAGIC * Linear scaling with complaint volume
# MAGIC * Partitioned inference for large datasets
# MAGIC * Can process millions of complaints with same code
# MAGIC
# MAGIC **Maintainability:**
# MAGIC * Feature list matches training (71 features)
# MAGIC * Model versioning through Unity Catalog
# MAGIC * Automated null handling and preprocessing
# MAGIC * Clear separation of concerns (inference → aggregation → output)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🏆 Pipeline Complete:**
# MAGIC * Bronze → Silver → Gold → **Ward Risk Scores** ✅
# MAGIC * ML models trained and deployed ✅
# MAGIC * Operational analytics table ready ✅
# MAGIC * Dashboard-ready output format ✅
# MAGIC
# MAGIC **📊 Output Table:** `civic_lens.output.bangalore_ward_risk`  
# MAGIC **🎯 Use Cases:** Resource allocation, alerting, dashboards, policy evaluation  
# MAGIC **♻️ Refresh:** Recommended daily or weekly  
# MAGIC
# MAGIC ---
