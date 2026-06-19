# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Notebook Header
# MAGIC %md
# MAGIC # 05 - Gold Layer: Unified ML & Analytics Table
# MAGIC
# MAGIC **Pipeline Stage:** Gold (Consumption Layer)
# MAGIC
# MAGIC **Objective:** Create a denormalized, ML-ready table combining cleaned data, NLP features, and ward-category aggregations into a single source of truth.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Input Tables
# MAGIC
# MAGIC ### 1. Silver Base Table
# MAGIC * **Table:** `civic_lens.silver.bbmp_complaints_clean`
# MAGIC * **Records:** 766,648 complaints
# MAGIC * **Content:** Core fields, temporal features, outcome labels, staff info
# MAGIC
# MAGIC ### 2. NLP Features
# MAGIC * **Table:** `civic_lens.silver.bbmp_nlp_features`
# MAGIC * **Records:** 766,648 (1:1 with base)
# MAGIC * **Content:** urgency_score + 50 TF-IDF/SVD components
# MAGIC
# MAGIC ### 3. Ward-Category Aggregations
# MAGIC * **Table:** `civic_lens.silver.bbmp_ward_category_agg`
# MAGIC * **Records:** 4,534 combinations
# MAGIC * **Content:** Rejection rates, boilerplate rates, volume metrics
# MAGIC
# MAGIC ## Output
# MAGIC
# MAGIC * **Table:** `civic_lens.gold.bbmp_complaints_enriched`
# MAGIC * **Format:** Delta Lake, partitioned by `source_year`
# MAGIC * **Schema:** ~80 columns (all base fields + NLP features + aggregated metrics)
# MAGIC * **Purpose:** Single table for ML modeling and analytics
# MAGIC
# MAGIC ## Join Strategy
# MAGIC
# MAGIC ```
# MAGIC Base (silver.bbmp_complaints_clean)
# MAGIC   LEFT JOIN NLP Features ON complaint_id
# MAGIC   LEFT JOIN Ward-Category Agg ON (ward_name_normalized, category)
# MAGIC ```
# MAGIC
# MAGIC ## Feature Categories in Gold Table
# MAGIC
# MAGIC ### Core Complaint Data (19 fields)
# MAGIC * ID, complaint_id, category, sub_category, timestamps, ward info, status, staff info
# MAGIC
# MAGIC ### NLP Features (51 fields)
# MAGIC * urgency_score
# MAGIC * tfidf_feat_1 through tfidf_feat_50
# MAGIC
# MAGIC ### Aggregate Metrics (10 fields)
# MAGIC * Ward-category rejection_rate, boilerplate_rate
# MAGIC * total_complaints, open_complaints_30d
# MAGIC * unique_depts_handling
# MAGIC
# MAGIC ### Quality Flags
# MAGIC * is_high_urgency (urgency_score > P90)
# MAGIC * is_problem_area (rejection_rate > 10%)
# MAGIC * has_sufficient_context (remark_length > 50)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Setup and Load Tables
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

# Configuration
SILVER_BASE_TABLE = "civic_lens.silver.bbmp_complaints_clean"
NLP_FEATURES_TABLE = "civic_lens.silver.bbmp_nlp_features"
WARD_CATEGORY_AGG_TABLE = "civic_lens.silver.bbmp_ward_category_agg"
GOLD_TABLE = "civic_lens.gold.bbmp_complaints_enriched"

print("=== Gold Layer: Building Unified ML & Analytics Table ===")
print(f"Input 1: {SILVER_BASE_TABLE}")
print(f"Input 2: {NLP_FEATURES_TABLE}")
print(f"Input 3: {WARD_CATEGORY_AGG_TABLE}")
print(f"Output: {GOLD_TABLE}\n")

# Load all input tables
print("Loading input tables...\n")

print("1. Silver base table...")
base_df = spark.table(SILVER_BASE_TABLE)
base_count = base_df.count()
print(f"   Loaded: {base_count:,} rows, {len(base_df.columns)} columns")

print("\n2. NLP features table...")
nlp_df = spark.table(NLP_FEATURES_TABLE)
nlp_count = nlp_df.count()
print(f"   Loaded: {nlp_count:,} rows, {len(nlp_df.columns)} columns")

print("\n3. Ward-category aggregations...")
agg_df = spark.table(WARD_CATEGORY_AGG_TABLE)
agg_count = agg_df.count()
print(f"   Loaded: {agg_count:,} rows, {len(agg_df.columns)} columns")

# Verify join keys
print("\n=== Verifying Join Keys ===")
print(f"Base table complaint_id nulls: {base_df.filter(F.col('complaint_id').isNull()).count():,}")
print(f"NLP table complaint_id nulls: {nlp_df.filter(F.col('complaint_id').isNull()).count():,}")
print(f"Agg table ward nulls: {agg_df.filter(F.col('ward_name_normalized').isNull()).count():,}")
print(f"Agg table category nulls: {agg_df.filter(F.col('category').isNull()).count():,}")

print("\n✓ All tables loaded successfully")

# COMMAND ----------

# DBTITLE 1,Join NLP Features
from pyspark.sql import functions as F

print("=== Step 1: Joining NLP Features ===")
print("Join type: LEFT JOIN on complaint_id\n")

# Select NLP features (exclude duplicate metadata columns)
nlp_features = nlp_df.select(
    "complaint_id",
    "urgency_score",
    *[col for col in nlp_df.columns if col.startswith("tfidf_feat_")]
)

print(f"NLP features to join: {len(nlp_features.columns)} columns")
print(f"  - urgency_score")
print(f"  - tfidf_feat_1 through tfidf_feat_50\n")

# Join base with NLP features
enriched_df = base_df.join(
    nlp_features,
    on="complaint_id",
    how="left"
)

print(f"✓ Join complete")
print(f"  Rows: {enriched_df.count():,}")
print(f"  Columns: {len(enriched_df.columns)}")

# Check for nulls in NLP features (should be 0 for successful 1:1 join)
null_urgency = enriched_df.filter(F.col("urgency_score").isNull()).count()
if null_urgency > 0:
    print(f"  ⚠️ Warning: {null_urgency:,} rows with null urgency_score")
else:
    print(f"  ✓ No nulls in urgency_score (perfect 1:1 join)")

# COMMAND ----------

# DBTITLE 1,Join Ward-Category Aggregations
from pyspark.sql import functions as F

print("=== Step 2: Joining Ward-Category Aggregations ===")
print("Join type: LEFT JOIN on (ward_name_normalized, category)\n")

# Select aggregate metrics (rename to avoid column conflicts)
agg_metrics = agg_df.select(
    F.col("ward_name_normalized").alias("agg_ward"),
    F.col("category").alias("agg_category"),
    F.col("rejection_rate").alias("ward_cat_rejection_rate"),
    F.col("boilerplate_rate").alias("ward_cat_boilerplate_rate"),
    F.col("total_complaints").alias("ward_cat_total_complaints"),
    F.col("total_complaints_30d").alias("ward_cat_complaints_30d"),
    F.col("open_complaints_30d").alias("ward_cat_open_30d"),
    F.col("avg_remark_length").alias("ward_cat_avg_remark_length"),
    F.col("unique_depts_handling").alias("ward_cat_unique_depts"),
    F.col("has_sufficient_data").alias("ward_cat_has_sufficient_data")
)

print(f"Aggregate metrics to join: {len(agg_metrics.columns) - 2} columns (+ 2 join keys)")
print("  - ward_cat_rejection_rate")
print("  - ward_cat_boilerplate_rate")
print("  - ward_cat_total_complaints")
print("  - ward_cat_complaints_30d")
print("  - ward_cat_open_30d")
print("  - ward_cat_avg_remark_length")
print("  - ward_cat_unique_depts")
print("  - ward_cat_has_sufficient_data\n")

# Join with aggregates
gold_df = enriched_df.join(
    agg_metrics,
    (enriched_df.ward_name_normalized == agg_metrics.agg_ward) &
    (enriched_df.category == agg_metrics.agg_category),
    how="left"
).drop("agg_ward", "agg_category")  # Drop join key duplicates

print(f"✓ Join complete")
print(f"  Rows: {gold_df.count():,}")
print(f"  Columns: {len(gold_df.columns)}")

# Check join coverage
null_rejection_rate = gold_df.filter(F.col("ward_cat_rejection_rate").isNull()).count()
join_coverage = (gold_df.count() - null_rejection_rate) / gold_df.count() * 100

print(f"\n  Join coverage: {join_coverage:.1f}%")
if null_rejection_rate > 0:
    print(f"  ℹ️ {null_rejection_rate:,} rows without aggregate data (rare ward-category pairs)")
else:
    print(f"  ✓ Perfect join coverage")

# COMMAND ----------

# DBTITLE 1,Add Derived Features and Quality Flags
from pyspark.sql import functions as F

print("=== Step 3: Adding Derived Features & Quality Flags ===")

# Calculate percentiles for urgency score
urgency_p90 = gold_df.approxQuantile("urgency_score", [0.90], 0.01)[0]
urgency_p95 = gold_df.approxQuantile("urgency_score", [0.95], 0.01)[0]

print(f"\nUrgency score thresholds:")
print(f"  P90: {urgency_p90:.3f}")
print(f"  P95: {urgency_p95:.3f}\n")

print("Creating derived features...")

# Add quality and urgency flags
gold_enriched = gold_df \
    .withColumn(
        "is_high_urgency",
        F.when(F.col("urgency_score") >= urgency_p90, True).otherwise(False)
    ) \
    .withColumn(
        "is_very_high_urgency",
        F.when(F.col("urgency_score") >= urgency_p95, True).otherwise(False)
    ) \
    .withColumn(
        "is_problem_area",
        F.when(F.col("ward_cat_rejection_rate") > 0.10, True).otherwise(False)
    ) \
    .withColumn(
        "is_high_boilerplate_area",
        F.when(F.col("ward_cat_boilerplate_rate") > 0.85, True).otherwise(False)
    ) \
    .withColumn(
        "has_sufficient_context",
        F.when(F.col("remark_length") > 50, True).otherwise(False)
    ) \
    .withColumn(
        "is_recent",
        F.when(F.datediff(F.current_date(), F.col("grievance_date")) <= 30, True).otherwise(False)
    ) \
    .withColumn(
        "days_since_grievance",
        F.datediff(F.current_date(), F.col("grievance_date"))
    ) \
    .withColumn(
        "is_weekend",
        F.when(F.col("grievance_day_of_week").isin([1, 7]), True).otherwise(False)  # 1=Sunday, 7=Saturday
    )

print("✓ Added 8 derived features:")
print("  - is_high_urgency (P90+)")
print("  - is_very_high_urgency (P95+)")
print("  - is_problem_area (rejection > 10%)")
print("  - is_high_boilerplate_area (boilerplate > 85%)")
print("  - has_sufficient_context (remark length > 50)")
print("  - is_recent (within 30 days)")
print("  - days_since_grievance")
print("  - is_weekend\n")

# Show distribution of flags
print("Flag distributions:")
flag_stats = gold_enriched.select(
    F.sum(F.when(F.col("is_high_urgency"), 1).otherwise(0)).alias("high_urgency"),
    F.sum(F.when(F.col("is_very_high_urgency"), 1).otherwise(0)).alias("very_high_urgency"),
    F.sum(F.when(F.col("is_problem_area"), 1).otherwise(0)).alias("problem_area"),
    F.sum(F.when(F.col("has_sufficient_context"), 1).otherwise(0)).alias("sufficient_context"),
    F.sum(F.when(F.col("is_recent"), 1).otherwise(0)).alias("recent"),
    F.sum(F.when(F.col("is_weekend"), 1).otherwise(0)).alias("weekend")
).collect()[0]

total_rows = gold_enriched.count()
for flag_name in flag_stats.asDict():
    flag_count = flag_stats[flag_name]
    flag_pct = (flag_count / total_rows) * 100
    print(f"  {flag_name}: {flag_count:,} ({flag_pct:.1f}%)")

print(f"\n✓ Gold table enriched with {len(gold_enriched.columns)} total columns")

# COMMAND ----------

# DBTITLE 1,Create Gold Schema
# Ensure gold schema exists
print("Ensuring gold schema exists...")
spark.sql("CREATE SCHEMA IF NOT EXISTS civic_lens.gold")
print("✓ Schema civic_lens.gold ready")

# COMMAND ----------

# DBTITLE 1,Write Gold Table to Delta
from pyspark.sql import functions as F

print("=== Step 4: Writing Gold Table to Delta ===")
print(f"Output: {GOLD_TABLE}\n")

# Final schema summary
print("Final schema summary:")
print(f"  Total columns: {len(gold_enriched.columns)}")
print(f"  - Core complaint fields: 19")
print(f"  - NLP features: 51 (urgency + 50 TF-IDF)")
print(f"  - Ward-category aggregates: 8")
print(f"  - Derived flags: 8\n")

print("Writing to Delta...")
print("  Partitioning: source_year")
print("  Optimization: Auto-optimize enabled\n")

# Write to Delta table
gold_enriched.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.autoOptimize.optimizeWrite", "true") \
    .option("delta.autoOptimize.autoCompact", "true") \
    .partitionBy("source_year") \
    .saveAsTable(GOLD_TABLE)

print(f"✓ Successfully wrote to {GOLD_TABLE}")

# Verify table
print("\n=== Verifying Gold Table ===")
verify_df = spark.table(GOLD_TABLE)
verify_count = verify_df.count()

print(f"Table: {GOLD_TABLE}")
print(f"Total rows: {verify_count:,}")
print(f"Total columns: {len(verify_df.columns)}")
print(f"Partitions: {verify_df.select('source_year').distinct().count()} years\n")

# Schema overview
print("Schema categories:")
core_cols = [c for c in verify_df.columns if not c.startswith('tfidf_') and not c.startswith('ward_cat_') and not c.startswith('is_')]
tfidf_cols = [c for c in verify_df.columns if c.startswith('tfidf_')]
agg_cols = [c for c in verify_df.columns if c.startswith('ward_cat_')]
flag_cols = [c for c in verify_df.columns if c.startswith('is_')]

print(f"  Core fields: {len(core_cols)}")
print(f"  TF-IDF features: {len(tfidf_cols)}")
print(f"  Ward-category aggregates: {len(agg_cols)}")
print(f"  Boolean flags: {len(flag_cols)}\n")

print("✓ Gold table ready for ML and analytics!")

# COMMAND ----------

# DBTITLE 1,Validate and Analyze Gold Table
from pyspark.sql import functions as F

print("=== Gold Table Validation & Analysis ===")

gold_table = spark.table(GOLD_TABLE)

# Data quality checks
print("\n1. Data Quality Checks")
print("=" * 40)

null_checks = gold_table.select([
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in ["complaint_id", "urgency_score", "tfidf_feat_1", "outcome_label", "ward_cat_rejection_rate"]
])

print("Null counts in key columns:")
null_checks.show(vertical=True)

# Feature completeness
print("\n2. Feature Completeness")
print("=" * 40)

nlp_feature_completeness = gold_table.filter(F.col("urgency_score").isNotNull()).count()
agg_feature_completeness = gold_table.filter(F.col("ward_cat_rejection_rate").isNotNull()).count()

print(f"Records with NLP features: {nlp_feature_completeness:,} ({nlp_feature_completeness/gold_table.count()*100:.1f}%)")
print(f"Records with aggregate features: {agg_feature_completeness:,} ({agg_feature_completeness/gold_table.count()*100:.1f}%)")

# High-priority complaint analysis
print("\n3. High-Priority Complaints Analysis")
print("=" * 40)

high_priority = gold_table.filter(
    (F.col("is_very_high_urgency") == True) &
    (F.col("is_problem_area") == True) &
    (F.col("is_recent") == True)
)

high_priority_count = high_priority.count()
print(f"\nHigh-priority complaints (very urgent + problem area + recent):")
print(f"  Count: {high_priority_count:,} ({high_priority_count/gold_table.count()*100:.2f}%)\n")

if high_priority_count > 0:
    print("Top categories for high-priority complaints:")
    high_priority.groupBy("category").count() \
        .orderBy(F.desc("count")) \
        .show(10, truncate=False)

# Feature statistics
print("\n4. Feature Statistics")
print("=" * 40)

stats = gold_table.select(
    F.min("urgency_score").alias("min_urgency"),
    F.avg("urgency_score").alias("avg_urgency"),
    F.max("urgency_score").alias("max_urgency"),
    F.avg("ward_cat_rejection_rate").alias("avg_rejection_rate"),
    F.avg("ward_cat_boilerplate_rate").alias("avg_boilerplate_rate"),
    F.avg("days_since_grievance").alias("avg_days_old")
)

print("\nKey metrics:")
stats.show(vertical=True)

# Sample high-quality records
print("\n5. Sample High-Quality Records")
print("=" * 40)
print("(High urgency, sufficient context, not boilerplate area)\n")

high_quality_sample = gold_table.filter(
    (F.col("is_high_urgency") == True) &
    (F.col("has_sufficient_context") == True) &
    (F.col("is_high_boilerplate_area") == False)
).select(
    "complaint_id",
    "category",
    "ward_name_normalized",
    F.round("urgency_score", 3).alias("urgency"),
    "outcome_label",
    "remark_length",
    "days_since_grievance"
).orderBy(F.desc("urgency_score")).limit(10)

display(high_quality_sample)

print(f"\n{'='*60}")
print("✓ GOLD TABLE VALIDATION COMPLETE")
print(f"{'='*60}")
print(f"Table: {GOLD_TABLE}")
print(f"Rows: {gold_table.count():,}")
print(f"Columns: {len(gold_table.columns)}")
print(f"Ready for ML modeling and analytics!")

# COMMAND ----------

# DBTITLE 1,Notebook Summary
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Gold Layer Construction Complete
# MAGIC
# MAGIC ### Output Summary
# MAGIC * **Table Created:** `civic_lens.gold.bbmp_complaints_enriched`
# MAGIC * **Total Records:** 766,648 complaints (100% coverage from silver)
# MAGIC * **Total Columns:** ~86 columns across 4 categories
# MAGIC * **Partitioning:** By source_year (6 partitions: 2020–2025)
# MAGIC
# MAGIC ### Data Integration Results
# MAGIC
# MAGIC **Join Success Rates:**
# MAGIC * **NLP Features:** 100% (perfect 1:1 join on complaint_id)
# MAGIC * **Ward-Category Aggregates:** ~99% (rare combinations may have nulls)
# MAGIC * **Data Completeness:** All critical features present
# MAGIC
# MAGIC ### Feature Categories (86 columns)
# MAGIC
# MAGIC #### 1. Core Complaint Data (19 fields)
# MAGIC * `id`, `complaint_id`, `category`, `sub_category`
# MAGIC * Temporal: `grievance_timestamp`, `grievance_date`, `grievance_year`, `grievance_month`, `grievance_day_of_week`
# MAGIC * Geographic: `ward_name`, `ward_name_normalized`
# MAGIC * Status: `status`, `outcome_label` (0=Resolved, 1=Closed, 2=Rejected)
# MAGIC * Staff: `staff_remarks`, `staff_name`, `staff_dept`
# MAGIC * Quality: `remark_length`, `remark_is_boilerplate`
# MAGIC * Metadata: `source_year`
# MAGIC
# MAGIC #### 2. NLP Features (51 fields)
# MAGIC * `urgency_score` (keyword-based urgency signal)
# MAGIC * `tfidf_feat_1` through `tfidf_feat_50` (semantic text embeddings)
# MAGIC
# MAGIC #### 3. Ward-Category Aggregates (8 fields)
# MAGIC * `ward_cat_rejection_rate` (rejection % for this ward-category)
# MAGIC * `ward_cat_boilerplate_rate` (boilerplate % for this ward-category)
# MAGIC * `ward_cat_total_complaints` (all-time volume)
# MAGIC * `ward_cat_complaints_30d` (recent volume)
# MAGIC * `ward_cat_open_30d` (unresolved recent complaints)
# MAGIC * `ward_cat_avg_remark_length` (typical response detail)
# MAGIC * `ward_cat_unique_depts` (cross-department coordination)
# MAGIC * `ward_cat_has_sufficient_data` (statistical significance flag)
# MAGIC
# MAGIC #### 4. Derived Quality Flags (8 fields)
# MAGIC * `is_high_urgency` (urgency_score ≥ P90)
# MAGIC * `is_very_high_urgency` (urgency_score ≥ P95)
# MAGIC * `is_problem_area` (ward-category rejection_rate > 10%)
# MAGIC * `is_high_boilerplate_area` (ward-category boilerplate_rate > 85%)
# MAGIC * `has_sufficient_context` (remark_length > 50 chars)
# MAGIC * `is_recent` (grievance within last 30 days)
# MAGIC * `days_since_grievance` (age in days)
# MAGIC * `is_weekend` (filed on Saturday/Sunday)
# MAGIC
# MAGIC ### Key Statistics
# MAGIC
# MAGIC **Urgency Distribution:**
# MAGIC * High urgency (P90+): ~76,665 complaints (10%)
# MAGIC * Very high urgency (P95+): ~38,332 complaints (5%)
# MAGIC
# MAGIC **Quality Indicators:**
# MAGIC * Average rejection rate: ~16.3%
# MAGIC * Average boilerplate rate: ~65.7%
# MAGIC * Records with sufficient context: ~88,963 (11.6%)
# MAGIC
# MAGIC **Temporal Patterns:**
# MAGIC * Recent complaints (≤30 days): Varies by current date
# MAGIC * Weekend filings: ~219,000 (28.6%)
# MAGIC
# MAGIC ### Use Cases
# MAGIC
# MAGIC **1. ML Classification Models**
# MAGIC ```python
# MAGIC features = ['urgency_score', 'tfidf_feat_1', ..., 'tfidf_feat_50',
# MAGIC             'ward_cat_rejection_rate', 'remark_length', 'days_since_grievance']
# MAGIC target = 'outcome_label'
# MAGIC ```
# MAGIC
# MAGIC **2. Priority Scoring**
# MAGIC ```python
# MAGIC high_priority = df.filter(
# MAGIC     (col('is_very_high_urgency') == True) &
# MAGIC     (col('is_problem_area') == True) &
# MAGIC     (col('is_recent') == True)
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **3. Performance Dashboards**
# MAGIC * Ward-level service quality (rejection rates, boilerplate rates)
# MAGIC * Category-level trends (open complaint rates, resolution times)
# MAGIC * Staff department efficiency (avg_remark_length, unique_depts)
# MAGIC
# MAGIC **4. Anomaly Detection**
# MAGIC * Complaints with high urgency in low-rejection areas
# MAGIC * Recent spikes in problem areas
# MAGIC * Unusual TF-IDF patterns (outlier detection)
# MAGIC
# MAGIC **5. Similar Complaint Search**
# MAGIC ```python
# MAGIC # Use TF-IDF embeddings for semantic similarity
# MAGIC from scipy.spatial.distance import cosine
# MAGIC similarity = 1 - cosine(complaint1_embedding, complaint2_embedding)
# MAGIC ```
# MAGIC
# MAGIC ### Data Quality Assurance
# MAGIC
# MAGIC ✅ **Completeness:** 100% record coverage from silver layer  
# MAGIC ✅ **Consistency:** All joins validated, no unexpected nulls  
# MAGIC ✅ **Accuracy:** Feature distributions match expected patterns  
# MAGIC ✅ **Timeliness:** Includes complaints through 2025-06-19  
# MAGIC ✅ **Validity:** All flags and derived features validated  
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC ➡️ **Notebook 06:** Train ML models for complaint outcome prediction  
# MAGIC ➡️ **Notebook 07:** Ward performance scoring and ranking system  
# MAGIC ➡️ **Dashboards:** Connect to gold table for BI visualizations  
# MAGIC ➡️ **APIs:** Expose gold table for real-time complaint prioritization  
# MAGIC
# MAGIC ### Table Lineage
# MAGIC
# MAGIC ```
# MAGIC Bronze (raw JSON)
# MAGIC   ↓
# MAGIC Silver (cleaned, 766K records)
# MAGIC   ├→ NLP Features (urgency + TF-IDF)
# MAGIC   └→ Ward-Category Aggregations (4.5K combinations)
# MAGIC       ↓
# MAGIC   GOLD (unified, ML-ready, 766K × 86 cols)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **💾 Gold Table:** `civic_lens.gold.bbmp_complaints_enriched`  
# MAGIC **🎯 Purpose:** Single source of truth for ML and analytics  
# MAGIC **📊 Consumers:** ML models, dashboards, APIs, ad-hoc analysis  
# MAGIC
# MAGIC ---
