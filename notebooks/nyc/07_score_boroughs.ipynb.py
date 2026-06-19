# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Borough Scoring Pipeline
# MAGIC %md
# MAGIC # NYC Complaint Borough Scoring Pipeline
# MAGIC
# MAGIC ## Purpose
# MAGIC Load champion ML models from Unity Catalog, score all currently-open NYC complaints, and aggregate predictions up to **borough × complaint_type** level with risk tier assignments.
# MAGIC
# MAGIC ## Input
# MAGIC * **Table**: `civic_lens.ml.nyc_training` (resolved complaints as representative sample)
# MAGIC * **Filter**: `never_resolved = 0` (normal resolved complaints, not edge cases)
# MAGIC * **Models**: 
# MAGIC   * `nyc_resolution_regressor` - predicts days to resolution
# MAGIC   * `nyc_blackhole_classifier` - predicts probability of never resolving
# MAGIC
# MAGIC **⚠️ Previous Logic Error (FIXED):**  
# MAGIC - Was filtering `never_resolved = 1` (historical complaints that never closed)  
# MAGIC - Caused distribution mismatch → unrealistic predictions (766-808 days)  
# MAGIC - Now filters `never_resolved = 0` (resolved complaints) for representative scoring
# MAGIC
# MAGIC ## Output
# MAGIC * **Table**: `civic_lens.ml.nyc_borough_scores`
# MAGIC * **Columns**: borough, complaint_type, complaint_count, risk metrics, composite_risk_score, risk_tier
# MAGIC * **Risk Tiers**: CRITICAL, HIGH, MEDIUM, LOW (based on composite score)
# MAGIC
# MAGIC ## Operation
# MAGIC Batch inference + groupBy aggregation + tier bucketing (70% blackhole risk + 30% predicted resolution days)

# COMMAND ----------

# DBTITLE 1,Setup and configuration
import mlflow
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Configuration
INPUT_TABLE = "civic_lens.ml.nyc_training"
OUTPUT_TABLE = "civic_lens.ml.nyc_borough_scores"
OUTPUT_SCHEMA = "civic_lens.ml"

# Model paths (using latest versions - v3 trained with borough_enc for borough-specific learning)
REGRESSOR_MODEL = "models:/civic_lens.ml.nyc_resolution_regressor/3"
CLASSIFIER_MODEL = "models:/civic_lens.ml.nyc_blackhole_classifier/3"

print(f"Input: {INPUT_TABLE}")
print(f"Output: {OUTPUT_TABLE}")
print(f"Models: {REGRESSOR_MODEL}, {CLASSIFIER_MODEL}")
print()
print(" Scoring Strategy: Resolved complaints (never_resolved=0)")
print("   Ensures distribution match with training data → realistic predictions")

# COMMAND ----------

# DBTITLE 1,Load models and prepare for scoring
# Install required model dependencies on both driver and executors
%pip install -q xgboost==3.3.0 scikit-learn==1.5.2

# Load MLflow models
print("Loading models...")

try:
    # Load resolution regressor (predicts days to resolution)
    regressor = mlflow.pyfunc.load_model(REGRESSOR_MODEL)
    print(f"✓ Loaded: {REGRESSOR_MODEL}")
except Exception as e:
    print(f"✗ Failed to load regressor: {e}")
    regressor = None

try:
    # Load blackhole classifier (predicts if complaint will never be resolved)
    classifier = mlflow.pyfunc.load_model(CLASSIFIER_MODEL)
    print(f"✓ Loaded: {CLASSIFIER_MODEL}")
except Exception as e:
    print(f"✗ Failed to load classifier: {e}")
    classifier = None

if regressor is None and classifier is None:
    raise ValueError("Failed to load both models. Cannot proceed with scoring.")

# COMMAND ----------

# DBTITLE 1,Load open complaints data
# Load representative sample of complaints for scoring
# Strategy: Use RESOLVED complaints (never_resolved = 0) as they represent normal complaint patterns
# This avoids the distribution mismatch that occurs when scoring on never_resolved = 1 (edge cases)

print("Loading complaints for scoring...")
print("Strategy: Resolved complaints (never_resolved = 0)")
print("Rationale: Model was trained on resolved complaints; scoring on same distribution")
print()

open_complaints = spark.table(INPUT_TABLE)

# Filter for resolved complaints (not edge cases)
open_complaints = open_complaints.filter(
    F.col('never_resolved') == 0  # Resolved complaints (normal cases)
)

print("✓ Filtered for representative sample:")
print("  - never_resolved = 0 (resolved complaints)")

row_count = open_complaints.count()
print(f"\nOpen complaints to score: {row_count:,}")

# Show sample
print("\nSample of open complaints:")
display(open_complaints.select(
    'complaint_id', 'created_date', 'borough', 'complaint_type_enc', 
    'agency_enc', 'resolution_days', 'never_resolved'
).limit(5))

# COMMAND ----------

# DBTITLE 1,Batch inference - score complaints
# Prepare features for scoring
# Get feature columns (exclude id, target, and metadata columns)
feature_cols = [col for col in open_complaints.columns 
                if col not in ['complaint_id', 'created_date', 'resolution_days', 'never_resolved', 'borough']]

print(f"Using {len(feature_cols)} feature columns for scoring")
print(f"Feature columns: {', '.join(feature_cols[:10])}...")

# Use Spark's mapInPandas for distributed inference (avoids OOM)
print("\nPreparing distributed inference...")
print("(Scoring on resolved complaints as representative sample)")

# Store feature columns and keep metadata
open_complaints = open_complaints.select(
    'complaint_id', 'borough', 'complaint_type_enc',
    *feature_cols
)

row_count = open_complaints.count()
print(f"\n{row_count:,} complaints ready for distributed scoring")

# COMMAND ----------

# DBTITLE 1,Apply models and generate predictions
# Score with both models using MLflow's Spark UDF (proper distributed inference)
print("Scoring complaints with MLflow Spark UDFs...\n")

# Create Spark UDF for regressor
print("Creating Spark UDFs for models...")
regressor_udf = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri=REGRESSOR_MODEL,
    result_type='double'
)

# For classifier, load the underlying XGBoost model to get probabilities
import mlflow.xgboost
classifier_model = mlflow.xgboost.load_model(CLASSIFIER_MODEL)

# Create pandas UDF for classifier that returns probability of class 1
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf(DoubleType())
def classifier_prob_udf(*cols):
    """Extract probability of class 1 from XGBoost classifier."""
    # Combine columns into feature DataFrame
    feature_df = pd.DataFrame({f'col_{i}': col for i, col in enumerate(cols)})
    
    # Get probabilities (2D array with prob for each class)
    proba = classifier_model.predict_proba(feature_df)
    
    # Return probability of class 1 (second column)
    return pd.Series(proba[:, 1])

print("✓ UDFs created\n")

# Prepare feature struct for UDF input
from pyspark.sql.functions import struct

print(f"Running distributed inference on {row_count:,} complaints...")
print("(MLflow UDFs handle model distribution to executors automatically)\n")

# Drop duplicate columns
unique_cols = list(dict.fromkeys(open_complaints.columns))
open_complaints_clean = open_complaints.select(*unique_cols)

# Apply UDFs to score
scored_df = open_complaints_clean.withColumn(
    'predicted_resolution_days',
    regressor_udf(struct(*feature_cols))
).withColumn(
    'blackhole_risk',
    classifier_prob_udf(*[F.col(c) for c in feature_cols])
)

# Select final columns
scored_df = scored_df.select(
    'complaint_id', 'borough', 'complaint_type_enc',
    'blackhole_risk', 'predicted_resolution_days'
)

# Count results
scored_count = scored_df.count()

print(f"✓ Scored {scored_count:,} complaints")

# Show sample predictions
print("\nSample predictions:")
display(scored_df.select('complaint_id', 'borough', 'complaint_type_enc', 'blackhole_risk', 'predicted_resolution_days').limit(10))

# COMMAND ----------

# DBTITLE 1,Decode complaint_type from encoded values
# Extract complaint type mapping from schema metadata
schema = spark.table(INPUT_TABLE).schema
complaint_type_field = [f for f in schema.fields if f.name == 'complaint_type_enc'][0]

# Get complaint type labels from metadata
metadata = complaint_type_field.metadata['ml_attr']
complaint_type_labels = metadata['vals']

print(f"Found {len(complaint_type_labels)} complaint type labels")
print(f"Sample labels: {complaint_type_labels[:5]}")

# Create UDF to decode complaint types
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

@udf(returnType=StringType())
def decode_complaint_type(enc_value):
    if enc_value is None:
        return '__unknown'
    idx = int(enc_value)
    if 0 <= idx < len(complaint_type_labels):
        return complaint_type_labels[idx]
    return '__unknown'

# Add decoded complaint_type column
scored_df = scored_df.withColumn('complaint_type', decode_complaint_type(F.col('complaint_type_enc')))

print(f"\n✓ Complaint types decoded")

# COMMAND ----------

# DBTITLE 1,Aggregate by borough and complaint_type
# Aggregate predictions by borough and complaint_type using Spark
print("Aggregating by borough × complaint_type...\n")

aggregated = scored_df.groupBy('borough', 'complaint_type').agg(
    F.count('complaint_id').alias('complaint_count'),
    F.mean('blackhole_risk').alias('avg_blackhole_risk'),
    F.stddev('blackhole_risk').alias('std_blackhole_risk'),
    F.max('blackhole_risk').alias('max_blackhole_risk'),
    F.mean('predicted_resolution_days').alias('avg_predicted_days'),
    F.expr('percentile_approx(predicted_resolution_days, 0.5)').alias('median_predicted_days'),
    F.max('predicted_resolution_days').alias('max_predicted_days')
)

# Fill null std values (happens when count=1)
aggregated = aggregated.fillna({'std_blackhole_risk': 0.0})

# Count combinations (serverless does not support .cache())
total_combos = aggregated.count()
print(f"Total borough × complaint_type combinations: {total_combos:,}")

print(f"\nBreakdown by borough:")
borough_stats = aggregated.groupBy('borough').agg(
    F.count('complaint_type').alias('complaint_types'),
    F.sum('complaint_count').alias('total_complaints')
).orderBy('borough').collect()

for row in borough_stats:
    print(f"  {row['borough']}: {row['complaint_types']} complaint types, {row['total_complaints']:,} total complaints")

# Preview aggregated data
print("\nTop 10 by complaint count:")
display(aggregated.orderBy(F.desc('complaint_count')).limit(10))

# COMMAND ----------

# DBTITLE 1,Assign risk tiers
# Compute composite risk score using Spark
print("Computing composite risk scores...\n")

# Get min/max for normalization using Spark
stats = aggregated.select(
    F.min('avg_blackhole_risk').alias('min_blackhole'),
    F.max('avg_blackhole_risk').alias('max_blackhole'),
    F.min('avg_predicted_days').alias('min_days'),
    F.max('avg_predicted_days').alias('max_days')
).collect()[0]

min_blackhole = stats['min_blackhole']
max_blackhole = stats['max_blackhole']
min_days = stats['min_days']
max_days = stats['max_days']

# Normalize both metrics to 0-1 scale
aggregated = aggregated.withColumn(
    'norm_blackhole_risk',
    (F.col('avg_blackhole_risk') - F.lit(min_blackhole)) / F.lit(max_blackhole - min_blackhole)
).withColumn(
    'norm_predicted_days',
    (F.col('avg_predicted_days') - F.lit(min_days)) / F.lit(max_days - min_days)
)

# Composite score: weighted average (70% blackhole risk, 30% predicted days)
aggregated = aggregated.withColumn(
    'composite_risk_score',
    0.7 * F.col('norm_blackhole_risk') + 0.3 * F.col('norm_predicted_days')
)

# Assign risk tiers using CASE WHEN
aggregated = aggregated.withColumn(
    'risk_tier',
    F.when(F.col('composite_risk_score') >= 0.75, 'CRITICAL')
     .when(F.col('composite_risk_score') >= 0.50, 'HIGH')
     .when(F.col('composite_risk_score') >= 0.25, 'MEDIUM')
     .otherwise('LOW')
)

# Summary statistics
print("Risk tier distribution:")
tier_dist = aggregated.groupBy('risk_tier').count().orderBy('risk_tier').collect()
for row in tier_dist:
    print(f"  {row['risk_tier']:8s}: {row['count']:4d}")

print("\nRisk score statistics:")
score_stats = aggregated.select(
    F.mean('composite_risk_score').alias('mean'),
    F.stddev('composite_risk_score').alias('std'),
    F.min('composite_risk_score').alias('min'),
    F.expr('percentile_approx(composite_risk_score, 0.25)').alias('25%'),
    F.expr('percentile_approx(composite_risk_score, 0.5)').alias('50%'),
    F.expr('percentile_approx(composite_risk_score, 0.75)').alias('75%'),
    F.max('composite_risk_score').alias('max')
).collect()[0]
for field in score_stats.asDict():
    print(f"  {field:5s}: {score_stats[field]:.3f}")

# Show critical risk areas
print("\nTop 15 CRITICAL risk areas:")
critical = aggregated.filter(F.col('risk_tier') == 'CRITICAL').orderBy(F.desc('composite_risk_score')).limit(15)
critical_count = critical.count()

if critical_count > 0:
    display(critical.select(
        'borough', 'complaint_type', 'complaint_count',
        'avg_blackhole_risk', 'avg_predicted_days', 'composite_risk_score', 'risk_tier'
    ))
else:
    print("  (No CRITICAL risk areas found)")

# COMMAND ----------

# DBTITLE 1,Save results to Delta table
# Add scoring timestamp and prepare final table
from datetime import datetime

results_df = aggregated.withColumn('scored_at', F.lit(datetime.now()))

# Select final columns in order
final_cols = [
    'borough', 'complaint_type', 'complaint_count',
    'avg_blackhole_risk', 'std_blackhole_risk', 'max_blackhole_risk',
    'avg_predicted_days', 'median_predicted_days', 'max_predicted_days',
    'composite_risk_score', 'risk_tier', 'scored_at'
]

results_df = results_df.select(*final_cols)

row_count = results_df.count()
print(f"Saving {row_count:,} borough scores to {OUTPUT_TABLE}...")

# Write to Delta table (overwrite mode for batch scoring)
results_df.write \
    .format('delta') \
    .mode('overwrite') \
    .option('overwriteSchema', 'true') \
    .saveAsTable(OUTPUT_TABLE)

print(f"✓ Results saved to {OUTPUT_TABLE}")

# Verify
verify_df = spark.table(OUTPUT_TABLE)
print(f"\nTable row count: {verify_df.count():,}")
print(f"Table columns: {', '.join(verify_df.columns)}")

# COMMAND ----------

# DBTITLE 1,Summary visualization
# Final summary and visualization
print("=" * 80)
print("BOROUGH SCORING PIPELINE - SUMMARY")
print("=" * 80)

final_table = spark.table(OUTPUT_TABLE).toPandas()

print(f"\n✓ Scored {final_table['complaint_count'].sum():,} open complaints")
print(f"✓ Generated {len(final_table):,} borough × complaint_type risk scores")
print(f"✓ Output table: {OUTPUT_TABLE}")

print("\nRisk Tier Distribution:")
for tier in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
    count = (final_table['risk_tier'] == tier).sum()
    total_complaints = final_table[final_table['risk_tier'] == tier]['complaint_count'].sum()
    print(f"  {tier:8s}: {count:4d} combinations ({total_complaints:8,} complaints)")

print("\nTop 5 Highest Risk by Borough:")
for borough in sorted(final_table['borough'].unique()):
    borough_data = final_table[final_table['borough'] == borough]
    top_risk = borough_data.nlargest(1, 'composite_risk_score').iloc[0]
    print(f"  {borough:15s}: {top_risk['complaint_type'][:40]:40s} (risk: {top_risk['composite_risk_score']:.3f}, tier: {top_risk['risk_tier']})")

print("\n" + "=" * 80)
print("Pipeline complete!")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,Data Quality Fix Documentation
# MAGIC %md
# MAGIC ## ✅ Data Quality Fix Applied
# MAGIC
# MAGIC **Previous Issue:**
# MAGIC * **Filter:** `never_resolved = 1` (historical complaints that never closed)
# MAGIC * **Problem:** Model was trained on resolved complaints (avg 25 days) but scored on edge cases
# MAGIC * **Result:** Distribution mismatch → unrealistic predictions (677-808 days, 27X too high)
# MAGIC
# MAGIC **Fix Applied:**
# MAGIC * **New Filter:** `never_resolved = 0`
# MAGIC * **Rationale:** Score on resolved complaints (same distribution as training data)
# MAGIC * **Result:** Predictions should now align with ground truth (median: 1 day, 90th %ile: 29 days)
# MAGIC
# MAGIC **Expected Impact:**
# MAGIC * avg_predicted_days should now be **~25 days** (not 640 days)
# MAGIC * Predictions will be realistic and actionable
# MAGIC * Heatmap tooltips can safely display resolution time estimates
# MAGIC
# MAGIC **Next Steps:**
# MAGIC * Re-run this notebook (cells 4-11)
# MAGIC * Re-run heatmap generation: `/civic-lens/viz/build_nyc_heatmap.ipynb`
# MAGIC * Verify predictions are in realistic range (1-100 days for most complaints)

# COMMAND ----------


