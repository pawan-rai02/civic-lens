# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Notebook Header
# MAGIC %md
# MAGIC # 03 - Silver Aggregations: Ward & Category Analytics
# MAGIC
# MAGIC **Pipeline Stage:** Silver Aggregations (Analytics Layer)
# MAGIC
# MAGIC **Objective:** Create ward-category level aggregations for service quality metrics, complaint patterns, and performance KPIs.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Input
# MAGIC * **Table:** `civic_lens.silver.bbmp_complaints_clean`
# MAGIC * **Records:** ~766K cleaned complaints
# MAGIC * **Dimensions:** 199 wards × 32 categories
# MAGIC
# MAGIC ## Output
# MAGIC * **Table:** `civic_lens.silver.bbmp_ward_category_agg`
# MAGIC * **Format:** Delta Lake, partitioned by `category`
# MAGIC * **Grain:** One row per ward-category combination
# MAGIC * **Records:** 4,534 unique combinations
# MAGIC
# MAGIC ## Aggregation Metrics
# MAGIC
# MAGIC ### Quality Indicators
# MAGIC * **rejection_rate:** % of rejected/non-relevant complaints (quality issue)
# MAGIC * **boilerplate_rate:** % of generic automated responses (service quality)
# MAGIC * **avg_remark_length:** Average detail in staff responses
# MAGIC
# MAGIC ### Volume Metrics
# MAGIC * **total_complaints:** All-time complaint count
# MAGIC * **total_complaints_30d:** Last 30 days volume
# MAGIC * **open_complaints_30d:** Unresolved recent complaints
# MAGIC
# MAGIC ### Operational Metadata
# MAGIC * **unique_categories:** Complaint diversity
# MAGIC * **unique_depts_handling:** Cross-department coordination
# MAGIC * **has_sufficient_data:** Flag for statistical significance (≥10 complaints)
# MAGIC * **computation_timestamp:** Data freshness indicator
# MAGIC
# MAGIC ## Key Analyses
# MAGIC 1. **Problem Hotspots:** High rejection + high boilerplate rates
# MAGIC 2. **Best Performers:** Low rejection + detailed responses
# MAGIC 3. **Open Complaint Trends:** Recent unresolved issue patterns
# MAGIC 4. **Service Quality Benchmarking:** Cross-ward/category comparison
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Load Silver Table and Setup
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta

# Configuration
SILVER_TABLE = "civic_lens.silver.bbmp_complaints_clean"
OUTPUT_TABLE = "civic_lens.silver.bbmp_ward_category_agg"

print("=== Ward & Category Aggregation Pipeline ===")
print(f"Input: {SILVER_TABLE}")
print(f"Output: {OUTPUT_TABLE}\n")

# Load silver table
print("Loading silver table...")
silver_df = spark.table(SILVER_TABLE)

print(f"Total records: {silver_df.count():,}")
print(f"Columns: {len(silver_df.columns)}\n")

# Show data range
print("Data range:")
silver_df.select(
    F.min("grievance_date").alias("min_date"),
    F.max("grievance_date").alias("max_date")
).show()

# Quick stats
print("Quick stats:")
silver_df.select(
    F.countDistinct("ward_name_normalized").alias("unique_wards"),
    F.countDistinct("category").alias("unique_categories"),
    F.countDistinct("staff_dept").alias("unique_depts")
).show()

print("✓ Data loaded successfully")

# COMMAND ----------

# DBTITLE 1,Compute Ward-Level Metrics
from pyspark.sql import functions as F

print("=== Computing Ward-Level Aggregates ===")
print("Metrics: rejection_rate, boilerplate_rate\n")

# Ward-level aggregations
ward_agg = silver_df.groupBy("ward_name_normalized").agg(
    # Total complaints per ward
    F.count("*").alias("total_complaints"),
    
    # Rejection rate: (rejected complaints / total complaints with outcome)
    (F.sum(F.when(F.col("outcome_label") == 2, 1).otherwise(0)) / 
     F.sum(F.when(F.col("outcome_label").isNotNull(), 1).otherwise(0))).alias("ward_rejection_rate"),
    
    # Boilerplate rate: (boilerplate remarks / total remarks)
    (F.sum(F.when(F.col("remark_is_boilerplate") == True, 1).otherwise(0)) / 
     F.count("*")).alias("ward_boilerplate_rate"),
    
    # Additional useful metrics
    F.avg("remark_length").alias("avg_remark_length"),
    F.countDistinct("category").alias("unique_categories_in_ward")
)

print(f"Ward aggregates computed: {ward_agg.count():,} wards\n")

# Show top wards by rejection rate
print("Top 10 wards by rejection rate:")
ward_agg.filter(F.col("total_complaints") >= 100) \
    .orderBy(F.desc("ward_rejection_rate")) \
    .select(
        "ward_name_normalized",
        "total_complaints",
        F.round("ward_rejection_rate", 3).alias("rejection_rate"),
        F.round("ward_boilerplate_rate", 3).alias("boilerplate_rate")
    ) \
    .show(10, truncate=False)

# Show wards with lowest boilerplate rate (more detailed responses)
print("Top 10 wards by detailed responses (low boilerplate):")
ward_agg.filter(F.col("total_complaints") >= 100) \
    .orderBy("ward_boilerplate_rate") \
    .select(
        "ward_name_normalized",
        "total_complaints",
        F.round("ward_boilerplate_rate", 3).alias("boilerplate_rate"),
        F.round("avg_remark_length", 1).alias("avg_remark_len")
    ) \
    .show(10, truncate=False)

print("✓ Ward-level metrics computed")

# COMMAND ----------

# DBTITLE 1,Compute Category-Level Metrics (30-Day Window)
from pyspark.sql import functions as F
from datetime import datetime, timedelta

print("=== Computing Category-Level Aggregates ===")
print("Metric: open_complaints_30d (complaints without closed/resolved outcome in last 30 days)\n")

# Define 30-day cutoff from the most recent date in data
max_date = silver_df.agg(F.max("grievance_date")).collect()[0][0]
cutoff_date = max_date - timedelta(days=30)

print(f"Max date in data: {max_date}")
print(f"30-day cutoff: {cutoff_date}\n")

# Filter to last 30 days
recent_df = silver_df.filter(F.col("grievance_date") >= cutoff_date)

print(f"Records in last 30 days: {recent_df.count():,}\n")

# Category-level aggregation
# Open complaints = outcome_label is NULL (In Progress, Registered, ReOpen)
category_agg = recent_df.groupBy("category").agg(
    # Total complaints in last 30 days
    F.count("*").alias("total_complaints_30d"),
    
    # Open complaints (NULL outcome = not yet closed/resolved/rejected)
    F.sum(F.when(F.col("outcome_label").isNull(), 1).otherwise(0)).alias("category_open_complaints_30d"),
    
    # Closed complaints in last 30 days
    F.sum(F.when(F.col("outcome_label") == 1, 1).otherwise(0)).alias("closed_30d"),
    
    # Resolved complaints in last 30 days
    F.sum(F.when(F.col("outcome_label") == 0, 1).otherwise(0)).alias("resolved_30d"),
    
    # Rejected complaints in last 30 days
    F.sum(F.when(F.col("outcome_label") == 2, 1).otherwise(0)).alias("rejected_30d"),
    
    # Open rate
    (F.sum(F.when(F.col("outcome_label").isNull(), 1).otherwise(0)) / F.count("*")).alias("open_rate_30d")
)

print(f"Category aggregates computed: {category_agg.count():,} categories\n")

# Show categories with most open complaints
print("Categories with most open complaints (last 30 days):")
category_agg.orderBy(F.desc("category_open_complaints_30d")) \
    .select(
        "category",
        "total_complaints_30d",
        "category_open_complaints_30d",
        F.round("open_rate_30d", 3).alias("open_rate")
    ) \
    .show(15, truncate=False)

print("✓ Category-level metrics computed")

# COMMAND ----------

# DBTITLE 1,Compute Ward-Category Cross Aggregates
from pyspark.sql import functions as F
from datetime import timedelta

print("=== Computing Ward-Category Cross Aggregates ===")
print("Combining both dimensions for detailed analysis\n")

# Get the cutoff date for 30-day window
max_date = silver_df.agg(F.max("grievance_date")).collect()[0][0]
cutoff_date_30d = max_date - timedelta(days=30)

# Ward-Category cross aggregation
ward_category_agg = silver_df.groupBy("ward_name_normalized", "category").agg(
    # Overall metrics
    F.count("*").alias("total_complaints"),
    
    # Rejection rate (safe division to handle cases with no outcomes)
    F.when(
        F.sum(F.when(F.col("outcome_label").isNotNull(), 1).otherwise(0)) > 0,
        F.sum(F.when(F.col("outcome_label") == 2, 1).otherwise(0)) / 
        F.sum(F.when(F.col("outcome_label").isNotNull(), 1).otherwise(0))
    ).otherwise(0.0).alias("rejection_rate"),
    
    # Boilerplate rate
    (F.sum(F.when(F.col("remark_is_boilerplate") == True, 1).otherwise(0)) / 
     F.count("*")).alias("boilerplate_rate"),
    
    # 30-day open complaints for this ward-category combo
    F.sum(F.when(
        (F.col("grievance_date") >= cutoff_date_30d) & 
        (F.col("outcome_label").isNull()),
        1
    ).otherwise(0)).alias("open_complaints_30d"),
    
    # 30-day total complaints
    F.sum(F.when(
        F.col("grievance_date") >= cutoff_date_30d,
        1
    ).otherwise(0)).alias("total_complaints_30d"),
    
    # Additional context
    F.avg("remark_length").alias("avg_remark_length"),
    F.countDistinct("staff_dept").alias("unique_depts_handling")
)

print(f"Ward-Category cross aggregates: {ward_category_agg.count():,} combinations\n")

# Show top ward-category pairs by volume
print("Top 20 ward-category pairs by complaint volume:")
ward_category_agg.orderBy(F.desc("total_complaints")) \
    .select(
        "ward_name_normalized",
        "category",
        "total_complaints",
        "open_complaints_30d",
        F.round("rejection_rate", 3).alias("rej_rate"),
        F.round("boilerplate_rate", 3).alias("boiler_rate")
    ) \
    .show(20, truncate=False)

print("✓ Ward-category cross aggregates computed")

# COMMAND ----------

# DBTITLE 1,Write Aggregated Table to Delta
from pyspark.sql import functions as F

print(f"=== Writing Aggregated Data to {OUTPUT_TABLE} ===")

# Prepare final aggregated table
# Add computation timestamp for tracking
final_agg = ward_category_agg.withColumn(
    "computation_timestamp",
    F.current_timestamp()
)

# Add data quality metrics
final_agg = final_agg.withColumn(
    "has_sufficient_data",
    F.when(F.col("total_complaints") >= 10, True).otherwise(False)
)

print(f"\nFinal schema ({len(final_agg.columns)} columns):")
for col_name in final_agg.columns:
    print(f"  - {col_name}")

# Write to Delta with optimizations
print("\nWriting to Delta table...")
final_agg.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.autoOptimize.optimizeWrite", "true") \
    .option("delta.autoOptimize.autoCompact", "true") \
    .partitionBy("category") \
    .saveAsTable(OUTPUT_TABLE)

print(f"✓ Successfully wrote {final_agg.count():,} rows to {OUTPUT_TABLE}")

# Verify write
print("\n=== Verification ===")
written_table = spark.table(OUTPUT_TABLE)
print(f"Table row count: {written_table.count():,}")
print(f"Partitions: {written_table.select('category').distinct().count():,} categories")

print("\n✓ Aggregation pipeline complete!")

# COMMAND ----------

# DBTITLE 1,Validate and Analyze Aggregated Table
from pyspark.sql import functions as F

print("=== Aggregated Table Analysis ===")

agg_table = spark.table(OUTPUT_TABLE)

print(f"\nTotal ward-category combinations: {agg_table.count():,}")
print(f"Unique wards: {agg_table.select('ward_name_normalized').distinct().count():,}")
print(f"Unique categories: {agg_table.select('category').distinct().count():,}\n")

# Summary statistics
print("Summary Statistics:")
agg_table.select(
    F.sum("total_complaints").alias("total_complaints_all"),
    F.sum("open_complaints_30d").alias("total_open_30d"),
    F.avg("rejection_rate").alias("avg_rejection_rate"),
    F.avg("boilerplate_rate").alias("avg_boilerplate_rate")
).show()

# Top problematic ward-category pairs (high rejection, low quality responses)
print("\nTop 10 problematic combinations (high rejection + high boilerplate):")
agg_table.filter(
    (F.col("total_complaints") >= 50) & 
    (F.col("rejection_rate") > 0.05)
).withColumn(
    "problem_score",
    F.col("rejection_rate") + F.col("boilerplate_rate")
).orderBy(F.desc("problem_score")) \
.select(
    "ward_name_normalized",
    "category",
    "total_complaints",
    F.round("rejection_rate", 3).alias("rej_rate"),
    F.round("boilerplate_rate", 3).alias("boiler_rate"),
    "open_complaints_30d"
).show(10, truncate=False)

# Best performing ward-category pairs (low rejection, detailed responses)
print("\nTop 10 best performing combinations (low rejection + low boilerplate):")
agg_table.filter(
    (F.col("total_complaints") >= 100)
).withColumn(
    "quality_score",
    2.0 - (F.col("rejection_rate") + F.col("boilerplate_rate"))
).orderBy(F.desc("quality_score")) \
.select(
    "ward_name_normalized",
    "category",
    "total_complaints",
    F.round("rejection_rate", 3).alias("rej_rate"),
    F.round("boilerplate_rate", 3).alias("boiler_rate"),
    F.round("avg_remark_length", 1).alias("avg_remark_len")
).show(10, truncate=False)

# Categories with most open complaints
print("\nCategories with highest open complaint rates (last 30 days):")
agg_table.groupBy("category").agg(
    F.sum("total_complaints_30d").alias("total_30d"),
    F.sum("open_complaints_30d").alias("open_30d"),
    F.when(
        F.sum("total_complaints_30d") > 0,
        F.sum("open_complaints_30d") / F.sum("total_complaints_30d")
    ).otherwise(0.0).alias("open_rate")
).filter(F.col("total_30d") >= 50) \
.orderBy(F.desc("open_rate")) \
.select(
    "category",
    "total_30d",
    "open_30d",
    F.round("open_rate", 3).alias("open_rate")
).show(15, truncate=False)

print("\n✓ Aggregation analysis complete!")
print(f"\n📈 Aggregated table ready: {OUTPUT_TABLE}")

# COMMAND ----------

# DBTITLE 1,Notebook Summary
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Ward & Category Aggregation Complete
# MAGIC
# MAGIC ### Output Summary
# MAGIC * **Table Created:** `civic_lens.silver.bbmp_ward_category_agg`
# MAGIC * **Grain:** Ward × Category combinations
# MAGIC * **Total Combinations:** 4,534 unique pairs
# MAGIC * **Coverage:** 199 wards, 32 categories
# MAGIC
# MAGIC ### Key Insights
# MAGIC
# MAGIC **Overall Quality Metrics:**
# MAGIC * **Average Rejection Rate:** 16.3% (quality concern)
# MAGIC * **Average Boilerplate Rate:** 65.7% (automation dominance)
# MAGIC * **Open Complaints (30d):** 7,195 unresolved
# MAGIC
# MAGIC **Top Problem Areas (High Rejection + High Boilerplate):**
# MAGIC 1. **Sarakki - Town Planning:** 100% rejection, 99.1% boilerplate
# MAGIC 2. **J.P. Nagar - Town Planning:** 90% rejection, 93.2% boilerplate
# MAGIC 3. **Ramamurthy Nagar - Others:** 94.4% rejection, 71.3% boilerplate
# MAGIC
# MAGIC **Best Performing Areas (Low Rejection + Detailed Responses):**
# MAGIC 1. **Hagadooru - E khata/Khata:** 0% rejection, 2.5% boilerplate, 84.2 avg length
# MAGIC 2. **Varthur - E khata/Khata:** 0% rejection, 5.9% boilerplate, 86.8 avg length
# MAGIC 3. **Hemmigepura - Veterinary:** 1.4% rejection, 8.7% boilerplate, 142.7 avg length
# MAGIC
# MAGIC **Categories with Highest Open Rates (Last 30 Days):**
# MAGIC * **Others:** 100% open (396/396)
# MAGIC * **Road Infrastructure:** 91.8% open (551/600)
# MAGIC * **Storm Water Drain:** 87.1% open (269/309)
# MAGIC * **Town Planning:** 83.7% open (87/104)
# MAGIC
# MAGIC ### Use Cases
# MAGIC * **Resource Allocation:** Target high-volume, high-rejection areas
# MAGIC * **Training Needs:** Identify departments with high boilerplate rates
# MAGIC * **Performance Benchmarking:** Compare wards/categories against peers
# MAGIC * **Predictive Modeling:** Features for complaint outcome prediction
# MAGIC
# MAGIC ### Aggregation Schema (13 columns)
# MAGIC * **Dimensions:** `ward_name_normalized`, `category`
# MAGIC * **Quality:** `rejection_rate`, `boilerplate_rate`, `avg_remark_length`
# MAGIC * **Volume:** `total_complaints`, `total_complaints_30d`, `open_complaints_30d`
# MAGIC * **Metadata:** `unique_categories`, `unique_depts_handling`, `has_sufficient_data`, `computation_timestamp`
# MAGIC
# MAGIC ### Next Steps
# MAGIC ➡️ **Notebook 04:** NLP feature engineering on staff remarks (urgency score, TF-IDF/SVD embeddings)
# MAGIC
# MAGIC ---
