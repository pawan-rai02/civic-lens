# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Load and join tables to build Gold layer
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# Load tables
silver_df = spark.table("civic_lens.silver.nyc_311_cleaned")
agg_df = spark.table("civic_lens.silver.nyc_borough_agency_agg")
nlp_df = spark.table("civic_lens.silver.nyc_nlp_features")

# Get initial row count for verification
initial_count = silver_df.count()
print(f"Silver table rows: {initial_count:,}")

# Join with NLP features (1:1 on complaint_id)
gold_df = silver_df.join(
    nlp_df,
    on="complaint_id",
    how="left"
)

# Join with aggregates (many:1 on borough, agency, created_date)
gold_df = gold_df.join(
    agg_df,
    on=["borough", "agency", "created_date"],
    how="left"
)

# Verify no row fan-out
final_count = gold_df.count()
print(f"Gold table rows after joins: {final_count:,}")
assert final_count == initial_count, f"Row count mismatch! Expected {initial_count}, got {final_count}"

# Label encode categorical features
indexers = [
    StringIndexer(inputCol="complaint_type", outputCol="complaint_type_enc", handleInvalid="keep"),
    StringIndexer(inputCol="agency", outputCol="agency_enc", handleInvalid="keep"),
    StringIndexer(inputCol="borough", outputCol="borough_enc", handleInvalid="keep")
]

pipeline = Pipeline(stages=indexers)
indexer_model = pipeline.fit(gold_df)
gold_df = indexer_model.transform(gold_df)

# Select final feature set
feature_cols = [
    # Identifiers
    "complaint_id",
    "created_date",
    
    # Target variables
    "resolution_days",
    "never_resolved",
    
    # Time features
    "dow_filed",
    "hour_filed",
    "month_filed",
    
    # Location features
    "borough",
    "latitude",
    "longitude",
    
    # Categorical features (encoded)
    "complaint_type_enc",
    "agency_enc",
    "borough_enc",
    
    # Borough/Agency aggregates
    "borough_blackhole_rate",
    "agency_resolution_rate_hist",
    "agency_open_complaints_30d",
    
    # NLP features
    "urgency_score",
    "topic_id"
] + [f"tfidf_feat_{i}" for i in range(1, 51)]

gold_df = gold_df.select(feature_cols)

# Show sample
print("\nGold table sample:")
display(gold_df.limit(5))

# Write to Delta table
print("\nWriting to civic_lens.ml.nyc_training...")
gold_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("civic_lens.ml.nyc_training")

print(f"✓ Gold table created with {final_count:,} rows")
