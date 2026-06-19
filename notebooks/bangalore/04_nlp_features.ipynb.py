# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Notebook Header
# MAGIC %md
# MAGIC # 04 - NLP Feature Engineering: Staff Remarks Analysis
# MAGIC
# MAGIC **Pipeline Stage:** Feature Engineering (ML-Ready Features)
# MAGIC
# MAGIC **Objective:** Extract semantic features from staff remarks using NLP techniques to capture urgency signals and text embeddings for ML modeling.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Input
# MAGIC * **Table:** `civic_lens.silver.bbmp_complaints_clean`
# MAGIC * **Records:** 766,648 complaints with staff remarks
# MAGIC * **Text Field:** `staff_remarks` (average 15.7 chars, 88.4% boilerplate)
# MAGIC
# MAGIC ## Output
# MAGIC * **Table:** `civic_lens.silver.bbmp_nlp_features`
# MAGIC * **Format:** Delta Lake, partitioned by `category`
# MAGIC * **Features:** 57 columns total
# MAGIC   * 1 urgency score
# MAGIC   * 50 TF-IDF/SVD components
# MAGIC   * 6 metadata fields
# MAGIC
# MAGIC ## NLP Pipeline
# MAGIC
# MAGIC ### 1. Remark Diversity Analysis
# MAGIC * **Total Remarks:** 766,648
# MAGIC * **Distinct Remarks:** 37,817 (4.9% diversity)
# MAGIC * **Decision:** Skip LDA (diversity < 10% threshold)
# MAGIC * **Rationale:** TF-IDF/SVD more effective for high-duplication corpus
# MAGIC
# MAGIC ### 2. Urgency Score Computation
# MAGIC * **Method:** Keyword-based scoring with length normalization
# MAGIC * **Keywords:** 35 context-aware terms (urgent, emergency, health, sewage, pothole, etc.)
# MAGIC * **Formula:** `keyword_count / (log10(length + 1) + 1)`
# MAGIC * **Distribution:** Median=0.0, P90=0.34, Max=4.78
# MAGIC
# MAGIC ### 3. TF-IDF Vectorization
# MAGIC * **Training Strategy:** Smart sampling for quality
# MAGIC   * ALL 88,963 non-boilerplate remarks (diverse content)
# MAGIC   * 50,000 sampled boilerplate remarks (common vocabulary)
# MAGIC * **Configuration:**
# MAGIC   * Vocabulary: 5,000 terms
# MAGIC   * N-grams: (1, 2) unigrams + bigrams
# MAGIC   * Min_df: 5, Max_df: 0.8
# MAGIC   * Matrix sparsity: 99.85%
# MAGIC
# MAGIC ### 4. Dimensionality Reduction (SVD)
# MAGIC * **Components:** 50 (capturing 60.06% variance)
# MAGIC * **Algorithm:** Randomized (optimized for large matrices)
# MAGIC * **Top Component:** Explains 14.54% variance alone
# MAGIC
# MAGIC ### 5. Distributed Transformation
# MAGIC * **Method:** Spark `mapInPandas` for parallel processing
# MAGIC * **Partitions:** 32 (optimized for serverless)
# MAGIC * **Optimization:** Float32 encoding, memory-efficient batching
# MAGIC
# MAGIC ## Performance Optimizations
# MAGIC 1. **Smart Sampling:** Prioritize non-boilerplate for TF-IDF fitting
# MAGIC 2. **Distributed Processing:** 32-partition parallelism
# MAGIC 3. **Memory Management:** Float32 features, streaming batches
# MAGIC 4. **Serverless-Compatible:** No caching, auto-optimize writes
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Setup and Load Silver Data
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re

# Configuration
SILVER_TABLE = "civic_lens.silver.bbmp_complaints_clean"
OUTPUT_TABLE = "civic_lens.silver.bbmp_nlp_features"
N_TFIDF_COMPONENTS = 50  # TF-IDF -> SVD dimensions
MAX_FEATURES = 5000  # TF-IDF vocabulary size

print("=== NLP Feature Engineering Pipeline ===")
print(f"Input: {SILVER_TABLE}")
print(f"Output: {OUTPUT_TABLE}")
print(f"TF-IDF components: {N_TFIDF_COMPONENTS}\n")

# Load silver table
print("Loading silver table...")
silver_df = spark.table(SILVER_TABLE)

print(f"Total records: {silver_df.count():,}")
print(f"Columns: {len(silver_df.columns)}\n")

# Show sample remarks
print("Sample staff remarks:")
silver_df.select("complaint_id", "staff_remarks", "remark_length", "remark_is_boilerplate") \
    .filter(F.col("remark_length") > 50) \
    .show(5, truncate=70)

print("✓ Data loaded")

# COMMAND ----------

# DBTITLE 1,Analyze Remark Diversity for LDA Decision
from pyspark.sql import functions as F

print("=== Checking Remark Diversity for LDA ===")
print("LDA requires diverse text corpus. Checking distinct remarks...\n")

# Count total vs distinct remarks
total_remarks = silver_df.count()
distinct_remarks = silver_df.select("staff_remarks").distinct().count()

print(f"Total remarks: {total_remarks:,}")
print(f"Distinct remarks: {distinct_remarks:,}")
print(f"Diversity ratio: {distinct_remarks / total_remarks:.3%}\n")

# Check distribution of remark lengths
print("Remark length distribution:")
silver_df.select(
    F.min("remark_length").alias("min_len"),
    F.avg("remark_length").alias("avg_len"),
    F.max("remark_length").alias("max_len"),
    F.expr("percentile(remark_length, 0.5)").alias("median_len")
).show()

# Check boilerplate distribution
print("Boilerplate vs non-boilerplate:")
silver_df.groupBy("remark_is_boilerplate").count() \
    .withColumn("percentage", F.round(F.col("count") * 100.0 / total_remarks, 2)) \
    .orderBy(F.desc("count")) \
    .show()

# Most common remarks
print("Top 10 most common remarks:")
silver_df.groupBy("staff_remarks").count() \
    .orderBy(F.desc("count")) \
    .show(10, truncate=50)

# Decision on LDA
DIVERSITY_THRESHOLD = 0.10  # Need at least 10% unique remarks for LDA
use_lda = (distinct_remarks / total_remarks) >= DIVERSITY_THRESHOLD

print(f"\n{'=' * 60}")
if use_lda:
    print(f"✓ DECISION: Use LDA (diversity {distinct_remarks / total_remarks:.1%} >= {DIVERSITY_THRESHOLD:.0%})")
    print("  The corpus has sufficient diversity for topic modeling.")
else:
    print(f"✗ DECISION: Skip LDA (diversity {distinct_remarks / total_remarks:.1%} < {DIVERSITY_THRESHOLD:.0%})")
    print("  Too many duplicate remarks - TF-IDF/SVD will be more effective.")
print(f"{'=' * 60}")

# COMMAND ----------

# DBTITLE 1,Compute Urgency Score
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

print("=== Computing Urgency Score ===")
print("Urgency based on keyword presence in remarks\n")

# Define urgency keywords (adapt to Indian/Bangalore context)
urgency_keywords = [
    # High urgency
    "urgent", "emergency", "immediate", "critical", "danger", "hazard",
    "overflow", "leakage", "broken", "damage", "block", "accident",
    # Health/safety
    "health", "disease", "dengue", "malaria", "rats", "snake",
    "stagnant", "sewage", "waste", "smell", "stink",
    # Infrastructure urgency
    "collapse", "crack", "pothole", "cave", "fallen", "uprooted",
    # Public complaints
    "complaint", "complain", "request", "require", "need", "pending"
]

print(f"Urgency keywords ({len(urgency_keywords)}):")
print(f"  {', '.join(urgency_keywords[:15])}...\n")

# Create regex pattern for keyword matching
urgency_pattern = "|".join([f"(?i){kw}" for kw in urgency_keywords])

# Compute urgency score as count of matched keywords
# Normalize by remark length to avoid bias toward longer remarks
df_with_urgency = silver_df.withColumn(
    "urgency_keyword_count",
    F.size(F.split(F.lower(F.col("staff_remarks")), urgency_pattern)) - 1
).withColumn(
    "urgency_score",
    # Score = keywords / (log(length + 1) + 1) to normalize
    # +1 in denominator to avoid division by zero for very short remarks
    (F.col("urgency_keyword_count").cast(FloatType()) / 
     (F.log10(F.col("remark_length").cast(FloatType()) + 1) + 1))
)

print("Sample urgency scores:")
df_with_urgency.select(
    "complaint_id",
    "staff_remarks",
    "remark_length",
    "urgency_keyword_count",
    F.round("urgency_score", 3).alias("urgency_score")
).orderBy(F.desc("urgency_score")) \
 .show(10, truncate=60)

# Distribution of urgency scores
print("\nUrgency score statistics:")
df_with_urgency.select(
    F.min("urgency_score").alias("min"),
    F.avg("urgency_score").alias("avg"),
    F.max("urgency_score").alias("max"),
    F.expr("percentile(urgency_score, 0.5)").alias("median"),
    F.expr("percentile(urgency_score, 0.9)").alias("p90")
).show()

print("✓ Urgency scores computed")

# COMMAND ----------

# DBTITLE 1,Prepare Data for TF-IDF (Optimized Sampling)
from pyspark.sql import functions as F
import pandas as pd

print("=== Preparing Data for TF-IDF Fitting ===")
print("Optimization: Sample non-boilerplate remarks for fitting\n")

# Strategy: TF-IDF learns better from diverse, non-boilerplate text
# We'll sample heavily from non-boilerplate + some boilerplate for vocabulary coverage

total_count = df_with_urgency.count()
boilerplate_count = df_with_urgency.filter(F.col("remark_is_boilerplate") == True).count()
non_boilerplate_count = total_count - boilerplate_count

print(f"Total records: {total_count:,}")
print(f"Boilerplate: {boilerplate_count:,} ({boilerplate_count/total_count:.1%})")
print(f"Non-boilerplate: {non_boilerplate_count:,} ({non_boilerplate_count/total_count:.1%})\n")

# Sample strategy for fitting
# - Use ALL non-boilerplate remarks (they're more diverse)
# - Sample 10% of boilerplate remarks (for common words)
SAMPLE_SIZE_BOILERPLATE = min(50000, int(boilerplate_count * 0.1))

print(f"Sampling strategy for TF-IDF fitting:")
print(f"  Non-boilerplate: ALL {non_boilerplate_count:,} records")
print(f"  Boilerplate: {SAMPLE_SIZE_BOILERPLATE:,} records (sampled)")
print(f"  Total for fitting: ~{non_boilerplate_count + SAMPLE_SIZE_BOILERPLATE:,} records\n")

# Create training sample
non_boilerplate_df = df_with_urgency.filter(F.col("remark_is_boilerplate") == False)
boilerplate_sample = df_with_urgency.filter(F.col("remark_is_boilerplate") == True) \
    .sample(fraction=SAMPLE_SIZE_BOILERPLATE / boilerplate_count, seed=42)

train_df = non_boilerplate_df.union(boilerplate_sample)

print(f"Training sample size: {train_df.count():,}")
print("Collecting training remarks to pandas...")

# Collect to pandas for sklearn
# Select only necessary columns to minimize memory
train_pandas = train_df.select("complaint_id", "staff_remarks") \
    .toPandas()

print(f"✓ Collected {len(train_pandas):,} remarks for TF-IDF fitting")
print(f"  Memory usage: ~{train_pandas.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# COMMAND ----------

# DBTITLE 1,Fit TF-IDF and Truncated SVD
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re

print("=== Fitting TF-IDF Vectorizer + Truncated SVD ===")
print(f"Target: {N_TFIDF_COMPONENTS} SVD components from TF-IDF\n")

# Text preprocessing function
def preprocess_text(text):
    """Clean and normalize text for TF-IDF"""
    if pd.isna(text) or text == "":
        return ""
    # Lowercase
    text = str(text).lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Preprocessing remarks...")
train_pandas['remarks_clean'] = train_pandas['staff_remarks'].apply(preprocess_text)

# Remove empty remarks
train_pandas = train_pandas[train_pandas['remarks_clean'] != ""]
print(f"After cleaning: {len(train_pandas):,} non-empty remarks\n")

# Fit TF-IDF Vectorizer
print("Fitting TF-IDF vectorizer...")
print(f"  Max features: {MAX_FEATURES:,}")
print(f"  Min df: 5 (appears in at least 5 documents)")
print(f"  Max df: 0.8 (appears in at most 80% of documents)")

tfidf = TfidfVectorizer(
    max_features=MAX_FEATURES,
    min_df=5,  # Ignore very rare terms
    max_df=0.8,  # Ignore very common terms
    ngram_range=(1, 2),  # Unigrams and bigrams
    strip_accents='unicode',
    stop_words='english'  # Remove common English stop words
)

tfidf_matrix = tfidf.fit_transform(train_pandas['remarks_clean'])

print(f"✓ TF-IDF fitted")
print(f"  Vocabulary size: {len(tfidf.vocabulary_):,}")
print(f"  Matrix shape: {tfidf_matrix.shape}")
print(f"  Matrix sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2%}\n")

# Show top terms
feature_names = tfidf.get_feature_names_out()
print("Top 20 TF-IDF terms:")
for i, term in enumerate(feature_names[:20]):
    print(f"  {i+1}. {term}")

# Fit Truncated SVD for dimensionality reduction
print(f"\nFitting Truncated SVD ({N_TFIDF_COMPONENTS} components)...")
svd = TruncatedSVD(
    n_components=N_TFIDF_COMPONENTS,
    random_state=42,
    algorithm='randomized'  # Faster than 'arpack' for large matrices
)

tfidf_svd = svd.fit_transform(tfidf_matrix)

print(f"✓ SVD fitted")
print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
print(f"  Output shape: {tfidf_svd.shape}")

# Show explained variance by component
print("\nExplained variance by first 10 components:")
for i in range(min(10, N_TFIDF_COMPONENTS)):
    print(f"  Component {i+1}: {svd.explained_variance_ratio_[i]:.4f}")

print(f"\n✓ TF-IDF + SVD models ready for transformation")

# COMMAND ----------

# DBTITLE 1,Transform All Remarks (Batch Processing)
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType

print("=== Transforming All Remarks with TF-IDF + SVD ===")
print("Optimization: Using Spark's mapInPandas for distributed processing\n")

total_records = df_with_urgency.count()
print(f"Total records to transform: {total_records:,}\n")

# Define output schema for mapInPandas
output_schema = StructType([
    StructField("complaint_id", StringType(), False)
] + [
    StructField(f"tfidf_feat_{i+1}", FloatType(), False)
    for i in range(N_TFIDF_COMPONENTS)
])

# Function to transform a pandas batch (used by mapInPandas)
def transform_batch_udf(iterator):
    """
    Apply TF-IDF + SVD transformation to pandas DataFrame batches.
    This function is called by Spark's mapInPandas for each partition.
    """
    for pdf in iterator:
        if len(pdf) == 0:
            continue
            
        # Preprocess remarks
        pdf['remarks_clean'] = pdf['staff_remarks'].apply(preprocess_text)
        
        # Transform with TF-IDF
        tfidf_matrix = tfidf.transform(pdf['remarks_clean'])
        
        # Apply SVD
        svd_features = svd.transform(tfidf_matrix)
        
        # Create result DataFrame
        result_df = pdf[['complaint_id']].copy()
        
        # Add SVD features as columns (convert to float32 to save memory)
        for i in range(N_TFIDF_COMPONENTS):
            result_df[f'tfidf_feat_{i+1}'] = svd_features[:, i].astype(np.float32)
        
        yield result_df

# Transform using mapInPandas (distributed processing)
print("Transforming all remarks using distributed processing...")
print("(This may take several minutes for large datasets)\n")

# Repartition for optimal parallelism
NUM_PARTITIONS = 32  # Adjust based on cluster size

tfidf_features_spark = df_with_urgency \
    .select("complaint_id", "staff_remarks") \
    .repartition(NUM_PARTITIONS) \
    .mapInPandas(transform_batch_udf, schema=output_schema)

# Count records (triggers computation)
feature_count = tfidf_features_spark.count()

print(f"✓ Transformation complete")
print(f"  Total records: {feature_count:,}")
print(f"  Feature columns: {N_TFIDF_COMPONENTS} (tfidf_feat_1 to tfidf_feat_{N_TFIDF_COMPONENTS})")
print(f"  Partitions used: {NUM_PARTITIONS}\n")

# Show sample
print("Sample TF-IDF features:")
tfidf_features_spark.show(5)

# COMMAND ----------

# DBTITLE 1,Combine Features and Write to Delta
from pyspark.sql import functions as F

print("=== Creating Final Feature Table ===")
print(f"Output: {OUTPUT_TABLE}\n")

print(f"TF-IDF features ready: {tfidf_features_spark.count():,} rows\n")

# Join with urgency scores
print("Joining with urgency scores...")
final_features = df_with_urgency.select(
    "complaint_id",
    "urgency_score",
    "remark_is_boilerplate",
    "remark_length",
    "category",
    "ward_name_normalized",
    "grievance_date"
).join(
    tfidf_features_spark,
    on="complaint_id",
    how="inner"
)

print(f"Final feature table: {final_features.count():,} rows")
print(f"Total columns: {len(final_features.columns)}\n")

# Show schema
print("Schema:")
for col_name in final_features.columns[:15]:  # Show first 15 columns
    col_type = [f.dataType.simpleString() for f in final_features.schema.fields if f.name == col_name][0]
    print(f"  - {col_name}: {col_type}")
if len(final_features.columns) > 15:
    print(f"  ... and {len(final_features.columns) - 15} more TF-IDF feature columns")

# Show sample
print("\nSample records:")
final_features.select(
    "complaint_id",
    "urgency_score",
    "remark_is_boilerplate",
    "tfidf_feat_1",
    "tfidf_feat_2",
    "tfidf_feat_3"
).show(5)

# Feature statistics
print("\nFeature statistics:")
final_features.select(
    F.min("urgency_score").alias("min_urgency"),
    F.avg("urgency_score").alias("avg_urgency"),
    F.max("urgency_score").alias("max_urgency"),
    F.min("tfidf_feat_1").alias("min_tfidf_1"),
    F.avg("tfidf_feat_1").alias("avg_tfidf_1"),
    F.max("tfidf_feat_1").alias("max_tfidf_1")
).show()

# Write to Delta table
print(f"\nWriting to {OUTPUT_TABLE}...")
final_features.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.autoOptimize.optimizeWrite", "true") \
    .option("delta.autoOptimize.autoCompact", "true") \
    .partitionBy("category") \
    .saveAsTable(OUTPUT_TABLE)

print(f"✓ Successfully wrote to {OUTPUT_TABLE}")

# Verify
verify_table = spark.table(OUTPUT_TABLE)
print(f"\nVerification: {verify_table.count():,} rows in table")
print(f"Partitions: {verify_table.select('category').distinct().count()} categories\n")

print("✓ NLP feature engineering complete!")

# COMMAND ----------

# DBTITLE 1,Validate and Analyze NLP Features
from pyspark.sql import functions as F
import numpy as np

print("=== NLP Feature Validation & Analysis ===")

feature_table = spark.table(OUTPUT_TABLE)

print(f"\nTable: {OUTPUT_TABLE}")
print(f"Total records: {feature_table.count():,}")
print(f"Total columns: {len(feature_table.columns)}\n")

# Check for nulls
print("Null check:")
null_counts = feature_table.select([
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in ["complaint_id", "urgency_score", "tfidf_feat_1", "tfidf_feat_25", "tfidf_feat_50"]
])
null_counts.show()

# Urgency score distribution
print("\nUrgency score distribution:")
feature_table.select(
    F.expr("percentile(urgency_score, array(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0))").alias("percentiles")
).show(truncate=False)

# TF-IDF feature distributions
print("TF-IDF feature statistics (first 5 components):")
for i in range(1, 6):
    stats = feature_table.select(
        F.lit(f"tfidf_feat_{i}").alias("feature"),
        F.min(f"tfidf_feat_{i}").alias("min"),
        F.avg(f"tfidf_feat_{i}").alias("mean"),
        F.stddev(f"tfidf_feat_{i}").alias("std"),
        F.max(f"tfidf_feat_{i}").alias("max")
    ).collect()[0]
    print(f"  {stats['feature']}: min={stats['min']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, max={stats['max']:.4f}")

# Compare urgency scores by category
print("\nTop 10 categories by average urgency score:")
feature_table.groupBy("category").agg(
    F.count("*").alias("count"),
    F.avg("urgency_score").alias("avg_urgency"),
    F.stddev("urgency_score").alias("std_urgency")
).filter(F.col("count") >= 100) \
 .orderBy(F.desc("avg_urgency")) \
 .select(
     "category",
     "count",
     F.round("avg_urgency", 4).alias("avg_urgency"),
     F.round("std_urgency", 4).alias("std_urgency")
 ).show(10, truncate=False)

# Compare boilerplate vs non-boilerplate urgency
print("\nUrgency by boilerplate status:")
feature_table.groupBy("remark_is_boilerplate").agg(
    F.count("*").alias("count"),
    F.avg("urgency_score").alias("avg_urgency"),
    F.avg("tfidf_feat_1").alias("avg_tfidf_1"),
    F.avg("tfidf_feat_2").alias("avg_tfidf_2")
).show()

# Sample high-urgency complaints
print("\nSample high-urgency complaints (top 5):")
original_df = spark.table(SILVER_TABLE)
high_urgency = feature_table.orderBy(F.desc("urgency_score")).limit(5) \
    .join(original_df.select("complaint_id", "staff_remarks"), on="complaint_id", how="inner") \
    .select(
        "complaint_id",
        "category",
        F.round("urgency_score", 3).alias("urgency"),
        "staff_remarks"
    )

high_urgency.show(5, truncate=70)

print(f"\n{'='*70}")
print("✓ NLP FEATURE ENGINEERING COMPLETE")
print(f"{'='*70}")
print(f"Output table: {OUTPUT_TABLE}")
print(f"Features: urgency_score + {N_TFIDF_COMPONENTS} TF-IDF/SVD components")
print(f"Records: {feature_table.count():,}")
print(f"Ready for ML modeling and downstream analysis!")

# COMMAND ----------

# DBTITLE 1,Notebook Summary
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## ✅ NLP Feature Engineering Complete
# MAGIC
# MAGIC ### Output Summary
# MAGIC * **Table Created:** `civic_lens.silver.bbmp_nlp_features`
# MAGIC * **Total Records:** 766,648 complaints
# MAGIC * **Feature Dimensions:** 57 columns (1 urgency + 50 TF-IDF + 6 metadata)
# MAGIC * **Partitioning:** By category (32 partitions)
# MAGIC * **Data Quality:** 0 null values across all feature columns
# MAGIC
# MAGIC ### Feature Quality Metrics
# MAGIC
# MAGIC **Urgency Score Distribution:**
# MAGIC * **Median:** 0.0 (most remarks non-urgent)
# MAGIC * **90th Percentile:** 0.34
# MAGIC * **99th Percentile:** 0.59
# MAGIC * **Maximum:** 4.78
# MAGIC * **Non-Boilerplate Avg:** 0.219 (7.8× higher than boilerplate)
# MAGIC * **Boilerplate Avg:** 0.028
# MAGIC
# MAGIC **TF-IDF/SVD Features:**
# MAGIC * **Component 1:** mean=0.408, std=0.482 (high discriminative power)
# MAGIC * **Component 2:** mean=0.134, std=0.312
# MAGIC * **Total Variance Explained:** 60.06%
# MAGIC * **Feature Range:** Normalized embeddings in semantic space
# MAGIC
# MAGIC ### Top High-Urgency Categories
# MAGIC 1. **Call Center:** 0.202 avg urgency
# MAGIC 2. **Parks & Playgrounds:** 0.133 avg urgency
# MAGIC 3. **Others:** 0.102 avg urgency
# MAGIC 4. **Forest:** 0.093 avg urgency
# MAGIC 5. **Sanitation:** 0.092 avg urgency
# MAGIC
# MAGIC ### Technical Implementation
# MAGIC
# MAGIC **Model Training:**
# MAGIC * **Training Samples:** 138,887 remarks (~20.5 MB)
# MAGIC * **Vocabulary Size:** 5,000 terms
# MAGIC * **SVD Components:** 50
# MAGIC * **Explained Variance:** 60.06%
# MAGIC
# MAGIC **Distributed Processing:**
# MAGIC * **Method:** Spark mapInPandas
# MAGIC * **Partitions:** 32
# MAGIC * **Processing Time:** Serverless optimized
# MAGIC * **Memory Optimization:** Float32 encoding
# MAGIC
# MAGIC ### Feature Schema (57 columns)
# MAGIC
# MAGIC **Core Features:**
# MAGIC * `complaint_id` (string) - Primary key
# MAGIC * `urgency_score` (double) - Keyword-based urgency signal
# MAGIC * `tfidf_feat_1` through `tfidf_feat_50` (float) - Semantic embeddings
# MAGIC
# MAGIC **Metadata:**
# MAGIC * `remark_is_boilerplate` (boolean)
# MAGIC * `remark_length` (integer)
# MAGIC * `category` (string) - Partition key
# MAGIC * `ward_name_normalized` (string)
# MAGIC * `grievance_date` (date)
# MAGIC
# MAGIC ### ML-Ready Applications
# MAGIC
# MAGIC **Classification Tasks:**
# MAGIC * Complaint outcome prediction (resolved/closed/rejected)
# MAGIC * Priority/urgency classification
# MAGIC * Department routing automation
# MAGIC
# MAGIC **Clustering & Similarity:**
# MAGIC * Similar complaint discovery
# MAGIC * Topic clustering
# MAGIC * Anomaly detection
# MAGIC
# MAGIC **Recommendation Systems:**
# MAGIC * Staff assignment optimization
# MAGIC * Response template suggestions
# MAGIC
# MAGIC **Feature Engineering Downstream:**
# MAGIC * Combine with temporal features (notebook 02)
# MAGIC * Join with ward-category aggregates (notebook 03)
# MAGIC * Input for ensemble models (notebook 06)
# MAGIC
# MAGIC ### Optimization Highlights
# MAGIC 1. ✅ **Smart Sampling:** 100% non-boilerplate + 10% boilerplate = better TF-IDF
# MAGIC 2. ✅ **Distributed Processing:** 32-way parallelism for 766K records
# MAGIC 3. ✅ **Memory Efficient:** Float32 encoding, streaming transforms
# MAGIC 4. ✅ **Serverless Compatible:** No caching, optimized Delta writes
# MAGIC
# MAGIC ### Next Steps
# MAGIC ➡️ **Notebook 05:** Build gold layer with combined features for analytics
# MAGIC ➡️ **Notebook 06:** Train ML models using NLP features for outcome prediction
# MAGIC
# MAGIC ---
