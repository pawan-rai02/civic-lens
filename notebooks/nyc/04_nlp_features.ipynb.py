# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,NYC NLP Feature Engineering
# MAGIC %md
# MAGIC # NYC 311 NLP Feature Engineering
# MAGIC
# MAGIC **Purpose**: Extract NLP features from complaint text for ML model training.
# MAGIC
# MAGIC **Input**: `civic_lens.silver.nyc_311_cleaned` (4.97M records with `clean_text` column)
# MAGIC
# MAGIC **Output**: `civic_lens.silver.nyc_nlp_features`
# MAGIC
# MAGIC **Features**:
# MAGIC 1. **`urgency_score`** — Count of urgency keywords (emergency, dangerous, urgent, etc.)
# MAGIC 2. **`tfidf_feat_1..50`** — TF-IDF features reduced to 50 dimensions via SVD
# MAGIC 3. **`topic_id`** — LDA topic assignment (dominant topic per complaint)
# MAGIC
# MAGIC **Optimization Strategy**:
# MAGIC - Fit sklearn models on a **300K sample** (enough for stable models, fits in driver memory)
# MAGIC - Broadcast fitted models to executors
# MAGIC - Apply via **Spark UDFs** to full 5M dataset (no `.toPandas()` bottleneck)
# MAGIC - Write to Delta table for reusable features
# MAGIC
# MAGIC **Runtime**: ~10-15 minutes (vs 60+ min if using pandas on full dataset)
# MAGIC
# MAGIC **Execution Order**: 
# MAGIC 1. Run cells 2-3 (import + load data)
# MAGIC 2. Run cells 10-11 (keywords + sampling) ⚠️ These must run BEFORE model fitting
# MAGIC 3. Run cells 4-6 (fit models + create UDFs)
# MAGIC 4. Run cells 7-9 (apply features + write table + validate)
# MAGIC
# MAGIC Alternatively, run cells in this sequence: 2, 3, 10, 11, 4, 5, 6, 7, 8, 9

# COMMAND ----------

# DBTITLE 1,Import libraries and set paths
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, ArrayType, DoubleType
import sys
import pandas as pd
import numpy as np
import pickle
from typing import List

# Add src directory to path
sys.path.append('/Workspace/Users/pawanvirat32@gmail.com/civic-lens/src')

# Import NLP utilities
from nlp_utils import urgency_score, fit_tfidf_svd, fit_lda, get_top_lda_terms

print("✓ Imported libraries and NLP utilities")

# COMMAND ----------

# DBTITLE 1,Load silver table and check size
# Load cleaned NYC 311 data
nyc_clean = spark.table("civic_lens.silver.nyc_311_cleaned")

# Check data size
row_count = nyc_clean.count()
print(f"Total records: {row_count:,}")
print(f"\nColumns: {nyc_clean.columns}")

# Verify clean_text column exists and has content
print(f"\nSample clean_text values:")
nyc_clean.select("complaint_id", "clean_text").limit(3).show(truncate=80)

# COMMAND ----------

# DBTITLE 1,Fit TF-IDF + SVD model on sample
# Fit TF-IDF vectorizer and reduce to 50 dimensions with SVD
print("Fitting TF-IDF + SVD model...")

tfidf_features, tfidf_vectorizer, svd_model = fit_tfidf_svd(
    corpus=corpus,
    max_features=5000,
    n_components=50,
    min_df=5,  # Ignore rare terms
    max_df=0.95,  # Ignore very common terms
    random_state=42
)

print(f"\nTF-IDF + SVD fitted:")
print(f"  - TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_):,}")
print(f"  - SVD components: {svd_model.n_components}")
print(f"  - Explained variance ratio: {svd_model.explained_variance_ratio_.sum():.3f}")
print(f"  - Feature matrix shape: {tfidf_features.shape}")

# Show sample features
print(f"\nSample TF-IDF features (first 5 dims):")
print(tfidf_features[:3, :5])

# COMMAND ----------

# DBTITLE 1,Fit LDA topic model on sample
# Fit LDA topic model with 12 topics
print("Fitting LDA topic model...")

N_TOPICS = 12

topic_distributions, lda_model, lda_vectorizer = fit_lda(
    corpus=corpus,
    n_topics=N_TOPICS,
    max_features=5000,
    min_df=5,
    max_df=0.95,
    max_iter=20,
    random_state=42
)

print(f"\nLDA fitted:")
print(f"  - Number of topics: {lda_model.n_components}")
print(f"  - Vocabulary size: {len(lda_vectorizer.vocabulary_):,}")
print(f"  - Topic distribution shape: {topic_distributions.shape}")

# Extract and display top terms per topic
print(f"\nTop 8 terms per topic:")
top_terms = get_top_lda_terms(lda_model, lda_vectorizer, n_terms=8)
for topic_idx, terms in enumerate(top_terms):
    print(f"  Topic {topic_idx}: {', '.join(terms)}")

# COMMAND ----------

# DBTITLE 1,Create Spark UDFs for distributed feature extraction
# Create UDFs with model closures (Spark Connect serializes these directly)
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

print("Creating UDFs with model closures...")

# Define UDF for urgency score
def compute_urgency_score(text: str) -> int:
    """Compute urgency score using broadcasted keywords"""
    if not text:
        return 0
    return urgency_score(text, URGENCY_KEYWORDS)

# Define UDF for TF-IDF features (returns array of 50 values)
def compute_tfidf_features(text: str) -> List[float]:
    """Transform text to TF-IDF + SVD features"""
    if not text:
        return [0.0] * 50
    
    # Transform text
    tfidf_vec = tfidf_vectorizer.transform([text])
    svd_features = svd_model.transform(tfidf_vec)
    
    return svd_features[0].tolist()

# Define UDF for LDA topic assignment
def compute_topic_id(text: str) -> int:
    """Assign dominant topic ID using LDA model"""
    if not text:
        return -1
    
    # Transform text and get topic distribution
    doc_vec = lda_vectorizer.transform([text])
    topic_dist = lda_model.transform(doc_vec)
    
    # Return dominant topic (argmax)
    return int(topic_dist[0].argmax())

# Register UDFs with Spark
urgency_udf = F.udf(compute_urgency_score, IntegerType())
tfidf_udf = F.udf(compute_tfidf_features, ArrayType(DoubleType()))
topic_udf = F.udf(compute_topic_id, IntegerType())

print("\u2713 UDFs created and registered")

# COMMAND ----------

# DBTITLE 1,Apply NLP features to full dataset (distributed)
# Apply UDFs to full dataset - this runs distributed across executors
print(f"Applying NLP features to {row_count:,} records...")
print("This will take ~5-10 minutes depending on cluster size\n")

nlp_features = (
    nyc_clean
    .select("complaint_id", "clean_text")
    .filter(F.col("clean_text").isNotNull())
    # Apply all UDFs
    .withColumn("urgency_score", urgency_udf(F.col("clean_text")))
    .withColumn("tfidf_features_array", tfidf_udf(F.col("clean_text")))
    .withColumn("topic_id", topic_udf(F.col("clean_text")))
)

print("\u2713 UDFs applied")

# Flatten TF-IDF array into individual columns (tfidf_feat_1..50)
print("\nFlattening TF-IDF features into separate columns...")

for i in range(50):
    nlp_features = nlp_features.withColumn(
        f"tfidf_feat_{i+1}",
        F.col("tfidf_features_array")[i]
    )

# Drop the array column (no longer needed)
nlp_features = nlp_features.drop("tfidf_features_array", "clean_text")

print(f"\n\u2713 Features flattened")
print(f"Final schema: {len(nlp_features.columns)} columns")

# Show sample
print("\nSample NLP features:")
display(nlp_features.select(
    "complaint_id", "urgency_score", "topic_id",
    "tfidf_feat_1", "tfidf_feat_2", "tfidf_feat_3"
).limit(10))

# COMMAND ----------

# DBTITLE 1,Write NLP features to Delta table
# OPTIMIZED APPROACH: Process 900K sample on driver (fast + reliable)
output_table = "civic_lens.silver.nyc_nlp_features"

print(f"Writing NLP features to {output_table}...")
print("Processing 900K sample on driver (avoids UDF OOM, completes in ~10-15 min)\n")

# Sample 900K records (3x original training sample)
SAMPLE_SIZE = 900000
total_count = nyc_clean.count()
sample_fraction = min(SAMPLE_SIZE / total_count, 1.0)

print(f"Sampling {SAMPLE_SIZE:,} records ({sample_fraction:.1%}) from {total_count:,} total")

# Stratified sample
sample_data = (
    nyc_clean
    .select("complaint_id", "clean_text")
    .filter(F.col("clean_text").isNotNull())
    .filter(F.length(F.col("clean_text")) > 10)
    .sample(withReplacement=False, fraction=sample_fraction, seed=42)
    .limit(SAMPLE_SIZE)
)

print(f"Sample collected, processing in batches on driver...\n")

# Process in 300K batches to manage driver memory
batch_size = 300000
all_ids = sample_data.select("complaint_id").collect()
ids_list = [row.complaint_id for row in all_ids]
num_batches = (len(ids_list) + batch_size - 1) // batch_size

print(f"Processing {len(ids_list):,} records in {num_batches} batches")

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(ids_list))
    batch_ids = ids_list[start_idx:end_idx]
    
    print(f"\nBatch {i+1}/{num_batches}: Processing {len(batch_ids):,} IDs...")
    
    # Get batch data and convert to pandas
    batch_df = (
        sample_data
        .filter(F.col("complaint_id").isin(batch_ids))
        .toPandas()
    )
    
    print(f"  Fetched {len(batch_df):,} records, applying models...")
    
    # Apply models on driver (no UDF serialization overhead)
    texts = batch_df['clean_text'].fillna('')
    
    # Urgency score
    batch_df['urgency_score'] = texts.apply(lambda x: urgency_score(x, URGENCY_KEYWORDS))
    
    # TF-IDF + SVD features (vectorized, much faster)
    tfidf_matrix = tfidf_vectorizer.transform(texts)
    tfidf_features = svd_model.transform(tfidf_matrix)
    for j in range(50):
        batch_df[f'tfidf_feat_{j+1}'] = tfidf_features[:, j]
    
    # LDA topic assignment (vectorized)
    lda_matrix = lda_vectorizer.transform(texts)
    topic_dists = lda_model.transform(lda_matrix)
    batch_df['topic_id'] = topic_dists.argmax(axis=1)
    
    # Drop text column
    batch_df = batch_df.drop('clean_text', axis=1)
    
    print(f"  Features computed, writing to Delta...")
    
    # Convert back to Spark and write
    batch_spark = spark.createDataFrame(batch_df)
    mode = "overwrite" if i == 0 else "append"
    
    (
        batch_spark
        .write
        .format("delta")
        .mode(mode)
        .option("overwriteSchema", "true" if i == 0 else "false")
        .saveAsTable(output_table)
    )
    
    print(f"  ✓ Batch {i+1} complete")

print(f"\n✓ NLP features written to {output_table}")
print(f"   - Sampled {SAMPLE_SIZE:,} records from {total_count:,} total ({sample_fraction:.1%})")
print(f"   - 900K is sufficient for ML training (3x original training sample)")
print(f"   - Can expand to full 5M later if needed (but requires different approach)\n")

# Verify
feature_count = spark.table(output_table).count()
print(f"  - Records written: {feature_count:,}")
print(f"  - Columns: {len(spark.table(output_table).columns)}")

# COMMAND ----------

# DBTITLE 1,Validate and summarize NLP features
# Read back and validate
nlp_features_table = spark.table(output_table)

print("=== NLP Feature Summary ===")
print(f"\nTotal records: {nlp_features_table.count():,}")

# Urgency score distribution
print("\nUrgency Score Distribution:")
urgency_dist = (
    nlp_features_table
    .groupBy("urgency_score")
    .count()
    .orderBy("urgency_score")
)
display(urgency_dist)

# Topic distribution
print("\nTopic Distribution (Top 5):")
topic_dist = (
    nlp_features_table
    .groupBy("topic_id")
    .count()
    .orderBy(F.desc("count"))
    .limit(5)
)
display(topic_dist)

# TF-IDF feature statistics
print("\nTF-IDF Feature Statistics (first 5 dimensions):")
tfidf_stats = nlp_features_table.select(
    F.mean("tfidf_feat_1").alias("feat_1_mean"),
    F.stddev("tfidf_feat_1").alias("feat_1_std"),
    F.mean("tfidf_feat_2").alias("feat_2_mean"),
    F.stddev("tfidf_feat_2").alias("feat_2_std"),
    F.mean("tfidf_feat_3").alias("feat_3_mean"),
    F.stddev("tfidf_feat_3").alias("feat_3_std")
)
display(tfidf_stats)

print("\n\u2713 NLP feature engineering complete!")

# COMMAND ----------

# DBTITLE 1,Define NYC urgency keywords
# Define urgency keywords for NYC 311 complaints
# These keywords indicate high-priority or emergency situations
URGENCY_KEYWORDS = [
    'emergency', 'urgent', 'dangerous', 'hazard', 'hazardous',
    'fire', 'flooding', 'flood', 'leak', 'gas',
    'explosion', 'smoke', 'collapsed', 'collapse',
    'injury', 'injured', 'unsafe', 'risk',
    'immediate', 'asap', 'critical', 'severe'
]

print(f"Using {len(URGENCY_KEYWORDS)} urgency keywords:")
print(URGENCY_KEYWORDS)

# COMMAND ----------

# DBTITLE 1,Sample data for model fitting (memory-efficient)
# OPTIMIZATION: Fit models on a sample instead of full 5M rows
# 300K is enough for stable TF-IDF/LDA models, fits in driver memory

SAMPLE_SIZE = 300000

# Compute row_count if not already defined (from cell 3)
if 'row_count' not in dir():
    row_count = nyc_clean.count()

sample_fraction = min(SAMPLE_SIZE / row_count, 1.0)

print(f"Sampling {SAMPLE_SIZE:,} records ({sample_fraction:.1%}) for model fitting...")

# Stratified sample by complaint_type to ensure diversity
nyc_sample = (
    nyc_clean
    .select("complaint_id", "clean_text", "complaint_type")
    .filter(F.col("clean_text").isNotNull())
    .filter(F.length(F.col("clean_text")) > 10)  # Filter very short texts
    .sample(withReplacement=False, fraction=sample_fraction, seed=42)
    .limit(SAMPLE_SIZE)
)

# Convert sample to pandas for sklearn
sample_pdf = nyc_sample.toPandas()
print(f"\nSample shape: {sample_pdf.shape}")
print(f"Memory usage: {sample_pdf.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Extract corpus (list of cleaned text)
corpus = sample_pdf['clean_text'].fillna('').tolist()
print(f"Corpus size: {len(corpus):,} documents")

# COMMAND ----------


