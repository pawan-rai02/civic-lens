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
# MAGIC * **Table**: `civic_lens.ml.nyc_training` (rows where `never_resolved = 1`)
# MAGIC * **Models**: 
# MAGIC   * `nyc_resolution_regressor` - predicts days to resolution
# MAGIC   * `nyc_blackhole_classifier` - predicts probability of never resolving
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

# Model paths (using latest versions)
REGRESSOR_MODEL = "models:/civic_lens.ml.nyc_resolution_regressor/1"
CLASSIFIER_MODEL = "models:/civic_lens.ml.nyc_blackhole_classifier/1"

print(f"Input: {INPUT_TABLE}")
print(f"Output: {OUTPUT_TABLE}")
print(f"Models: {REGRESSOR_MODEL}, {CLASSIFIER_MODEL}")

# COMMAND ----------

# DBTITLE 1,Load models and prepare for scoring
# Install required model dependencies
import subprocess
import sys

print("Installing model dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost==3.3.0", "scikit-learn==1.5.2"])
print("✓ Dependencies installed\n")

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
# Load complaints where never_resolved = 1 (currently open)
# Note: If is_open column exists, filter by that instead

open_complaints = spark.table(INPUT_TABLE)

# Check if is_open column exists
if 'is_open' in open_complaints.columns:
    print("Filtering by is_open = True")
    open_complaints = open_complaints.filter(F.col('is_open') == True)
else:
    print("is_open column not found. Using never_resolved = 1 as proxy for open complaints")
    open_complaints = open_complaints.filter(F.col('never_resolved') == 1)

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

# Convert to pandas for MLflow model inference
# Note: For large datasets, consider batching or using spark_udf
print("\nConverting to pandas for inference (this may take a while for large datasets)...")

# Collect open complaints data
open_df = open_complaints.toPandas()

print(f"Loaded {len(open_df):,} rows for scoring")

# COMMAND ----------

# DBTITLE 1,Apply models and generate predictions
# Score with both models
print("Scoring complaints...\n")

# Prepare feature matrix
X = open_df[feature_cols]

# Apply blackhole classifier (probability of never being resolved)
if classifier:
    print("Running blackhole classifier...")
    blackhole_probs = classifier.predict(X)
    
    # If output is 2D (class probabilities), take probability of class 1
    if len(blackhole_probs.shape) > 1:
        open_df['blackhole_risk'] = blackhole_probs[:, 1]
    else:
        open_df['blackhole_risk'] = blackhole_probs
    print(f"✓ Blackhole risk scores added (mean: {open_df['blackhole_risk'].mean():.3f})")
else:
    open_df['blackhole_risk'] = 0.0
    print("⚠ Blackhole classifier not available, using 0.0")

# Apply resolution regressor (predicted days to resolution)
if regressor:
    print("\nRunning resolution regressor...")
    predicted_days = regressor.predict(X)
    open_df['predicted_resolution_days'] = predicted_days
    print(f"✓ Resolution predictions added (mean: {open_df['predicted_resolution_days'].mean():.1f} days)")
else:
    open_df['predicted_resolution_days'] = 0.0
    print("⚠ Resolution regressor not available, using 0.0")

# Display sample predictions
print("\nSample predictions:")
display(open_df[['complaint_id', 'borough', 'complaint_type_enc', 'blackhole_risk', 'predicted_resolution_days']].head(10))

# COMMAND ----------

# DBTITLE 1,Decode complaint_type from encoded values
# Extract complaint type mapping from schema metadata
schema = spark.table(INPUT_TABLE).schema
complaint_type_field = [f for f in schema.fields if f.name == 'complaint_type_enc'][0]

# Get complaint type labels from metadata (already a dict, no need to parse JSON)
metadata = complaint_type_field.metadata['ml_attr']
complaint_type_labels = metadata['vals']

print(f"Found {len(complaint_type_labels)} complaint type labels")
print(f"Sample labels: {complaint_type_labels[:5]}")

# Map encoded values back to complaint types
open_df['complaint_type'] = open_df['complaint_type_enc'].astype(int).map(
    lambda x: complaint_type_labels[x] if 0 <= x < len(complaint_type_labels) else '__unknown'
)

print(f"\nUnique complaint types in open data: {open_df['complaint_type'].nunique()}")

# COMMAND ----------

# DBTITLE 1,Aggregate by borough and complaint_type
# Aggregate predictions by borough and complaint_type
print("Aggregating by borough × complaint_type...\n")

aggregated = open_df.groupby(['borough', 'complaint_type']).agg({
    'complaint_id': 'count',
    'blackhole_risk': ['mean', 'std', 'max'],
    'predicted_resolution_days': ['mean', 'median', 'max']
}).reset_index()

# Flatten column names
aggregated.columns = [
    'borough', 'complaint_type', 'complaint_count',
    'avg_blackhole_risk', 'std_blackhole_risk', 'max_blackhole_risk',
    'avg_predicted_days', 'median_predicted_days', 'max_predicted_days'
]

# Fill NaN std values (happens when count=1)
aggregated['std_blackhole_risk'] = aggregated['std_blackhole_risk'].fillna(0)

print(f"Total borough × complaint_type combinations: {len(aggregated):,}")
print(f"\nBreakdown by borough:")
for borough in aggregated['borough'].unique():
    count = len(aggregated[aggregated['borough'] == borough])
    complaints = aggregated[aggregated['borough'] == borough]['complaint_count'].sum()
    print(f"  {borough}: {count} complaint types, {complaints:,} total complaints")

# Preview aggregated data
print("\nTop 10 by complaint count:")
display(aggregated.nlargest(10, 'complaint_count'))

# COMMAND ----------

# DBTITLE 1,Assign risk tiers
# Compute composite risk score
# Higher blackhole_risk + longer predicted_days = higher risk

print("Computing composite risk scores...\n")

# Normalize both metrics to 0-1 scale
from sklearn.preprocessing import MinMaxScaler

scaler_blackhole = MinMaxScaler()
scaler_days = MinMaxScaler()

aggregated['norm_blackhole_risk'] = scaler_blackhole.fit_transform(
    aggregated[['avg_blackhole_risk']]
)
aggregated['norm_predicted_days'] = scaler_days.fit_transform(
    aggregated[['avg_predicted_days']]
)

# Composite score: weighted average (70% blackhole risk, 30% predicted days)
aggregated['composite_risk_score'] = (
    0.7 * aggregated['norm_blackhole_risk'] + 
    0.3 * aggregated['norm_predicted_days']
)

# Assign risk tiers based on composite score
def assign_tier(score):
    if score >= 0.75:
        return 'CRITICAL'
    elif score >= 0.50:
        return 'HIGH'
    elif score >= 0.25:
        return 'MEDIUM'
    else:
        return 'LOW'

aggregated['risk_tier'] = aggregated['composite_risk_score'].apply(assign_tier)

# Summary statistics
print("Risk tier distribution:")
print(aggregated['risk_tier'].value_counts().sort_index())

print("\nRisk score statistics:")
print(aggregated['composite_risk_score'].describe())

# Show critical risk areas
print("\nTop 15 CRITICAL risk areas:")
critical = aggregated[aggregated['risk_tier'] == 'CRITICAL'].nlargest(15, 'composite_risk_score')
if len(critical) > 0:
    display(critical[[
        'borough', 'complaint_type', 'complaint_count', 
        'avg_blackhole_risk', 'avg_predicted_days', 'composite_risk_score', 'risk_tier'
    ]])
else:
    print("  (No CRITICAL risk areas found)")

# COMMAND ----------

# DBTITLE 1,Save results to Delta table
# Add scoring timestamp
from datetime import datetime
aggregated['scored_at'] = datetime.now()

# Select final columns
final_cols = [
    'borough', 'complaint_type', 'complaint_count',
    'avg_blackhole_risk', 'std_blackhole_risk', 'max_blackhole_risk',
    'avg_predicted_days', 'median_predicted_days', 'max_predicted_days',
    'composite_risk_score', 'risk_tier', 'scored_at'
]

results_df = aggregated[final_cols]

# Convert to Spark DataFrame
results_spark = spark.createDataFrame(results_df)

print(f"Saving {len(results_df):,} borough scores to {OUTPUT_TABLE}...")

# Write to Delta table (overwrite mode for batch scoring)
results_spark.write \
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


