# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Notebook Header
# MAGIC %md
# MAGIC # 02 - Silver Layer: Data Cleaning & Enrichment
# MAGIC
# MAGIC **Pipeline Stage:** Silver (Cleaned & Enriched Data)
# MAGIC
# MAGIC **Objective:** Transform raw bronze data into clean, analysis-ready silver layer with enriched features and quality checks.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Input
# MAGIC * **Table:** `civic_lens.bronze.bbmp_complaints_raw`
# MAGIC * **Records:** ~766K raw complaints
# MAGIC
# MAGIC ## Output
# MAGIC * **Table:** `civic_lens.silver.bbmp_complaints_clean`
# MAGIC * **Format:** Delta Lake, partitioned by `source_year`
# MAGIC * **Schema:** 19 columns (original + computed features)
# MAGIC
# MAGIC ## Transformations Applied
# MAGIC
# MAGIC ### 1. Temporal Features
# MAGIC * Parse `Grievance_Date` to proper timestamp
# MAGIC * Extract: `grievance_date`, `grievance_year`, `grievance_month`, `day_of_week`
# MAGIC
# MAGIC ### 2. Outcome Label Engineering
# MAGIC * **Class 0 (Resolved):** Successfully resolved complaints
# MAGIC * **Class 1 (Closed):** Closed or long-term solution
# MAGIC * **Class 2 (Rejected):** Rejected or non-relevant
# MAGIC * **NULL:** In-progress, registered, reopened
# MAGIC
# MAGIC ### 3. Staff Information Extraction
# MAGIC * Extract `staff_dept` from `Staff_Name` (format: "Name/Dept")
# MAGIC * Clean and normalize staff names
# MAGIC
# MAGIC ### 4. Remark Quality Features
# MAGIC * `remark_length`: Character count
# MAGIC * `remark_is_boilerplate`: Boolean flag for generic responses (<20 chars or common phrases)
# MAGIC
# MAGIC ### 5. Geographic Normalization
# MAGIC * Normalize ward names using `geo_utils.normalize_ward_name()`
# MAGIC * Lowercase, remove "ward" suffix, standardize formatting
# MAGIC
# MAGIC ### 6. Data Quality Checks
# MAGIC * Filter null `complaint_id` and `grievance_timestamp`
# MAGIC * Validate critical field completeness
# MAGIC * Report data quality metrics
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Silver Layer Transformations
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, TimestampType, BooleanType
import sys

# Add src directory to path for geo_utils import
sys.path.append("/Workspace/Users/pawanvirat32@gmail.com/civic-lens/src")
from geo_utils import normalize_ward_name

# Configuration
BRONZE_TABLE = "civic_lens.bronze.bbmp_complaints_raw"
SILVER_TABLE = "civic_lens.silver.bbmp_complaints_clean"

print("=== Silver Layer: Cleaning & Enrichment ===")
print(f"Input: {BRONZE_TABLE}")
print(f"Output: {SILVER_TABLE}\n")

# Read bronze table
print("Reading bronze table...")
bronze_df = spark.table(BRONZE_TABLE)
print(f"Bronze rows: {bronze_df.count():,}\n")

# Step 1: Parse Grievance_Date to proper timestamp
print("Step 1: Parsing Grievance_Date...")
df = bronze_df.withColumn(
    "grievance_timestamp",
    F.to_timestamp(F.col("Grievance_Date"), "yyyy-MM-dd HH:mm:ss.SSSSSSSSS")
)

# Extract date components for analysis
df = df.withColumn("grievance_date", F.to_date("grievance_timestamp")) \
       .withColumn("grievance_year", F.year("grievance_timestamp")) \
       .withColumn("grievance_month", F.month("grievance_timestamp")) \
       .withColumn("grievance_day_of_week", F.dayofweek("grievance_timestamp"))

print("  ✓ Parsed dates and extracted temporal features")

# Step 2: Build 3-class outcome_label from Grievance_Status
print("\nStep 2: Creating outcome_label (0=Resolved, 1=Closed, 2=Rejected)...")

# Status distribution check
status_counts = df.groupBy("Grievance_Status").count().orderBy(F.desc("count"))
print("\n  Status distribution:")
status_counts.show(10, truncate=False)

df = df.withColumn(
    "outcome_label",
    F.when(F.col("Grievance_Status") == "Resolved", 0)
     .when(F.col("Grievance_Status").isin(["Closed", "Long Term Solution"]), 1)
     .when(F.col("Grievance_Status").isin(["Rejected", "Non Relevant"]), 2)
     .otherwise(None)  # Registered, ReOpen, In Progress, null -> None
)

print("  ✓ Mapped statuses to outcome labels")

# Step 3: Extract staff_dept from Staff_Name (format: "Name/Dept")
print("\nStep 3: Extracting staff_dept from Staff_Name...")

df = df.withColumn(
    "staff_dept",
    F.when(
        F.col("Staff_Name").contains("/"),
        F.split(F.col("Staff_Name"), "/").getItem(1)
    ).otherwise(None)
)

df = df.withColumn(
    "staff_name_cleaned",
    F.when(
        F.col("Staff_Name").contains("/"),
        F.split(F.col("Staff_Name"), "/").getItem(0)
    ).otherwise(F.col("Staff_Name"))
)

print("  ✓ Extracted staff_dept and cleaned staff_name")

# Check unique departments
dept_counts = df.groupBy("staff_dept").count().orderBy(F.desc("count"))
print("\n  Top departments:")
dept_counts.show(10, truncate=False)

# Step 4: Compute remark_length and remark_is_boilerplate
print("\nStep 4: Computing remark features...")

df = df.withColumn("remark_length", F.length(F.col("Staff_Remarks")))

# Define common boilerplate phrases (case-insensitive)
boilerplate_patterns = [
    "attended",
    "closed",
    "work completed",
    "action taken",
    "done",
    "attended to",
    "complaint attended"
]

# Create regex pattern for boilerplate detection
boilerplate_regex = "|".join([f"(?i){pattern}" for pattern in boilerplate_patterns])

df = df.withColumn(
    "remark_is_boilerplate",
    F.when(
        (F.col("remark_length") <= 20) | 
        (F.col("Staff_Remarks").rlike(boilerplate_regex) & (F.col("remark_length") <= 50)),
        True
    ).otherwise(False)
)

print("  ✓ Computed remark_length and remark_is_boilerplate")

# Show boilerplate stats
boilerplate_stats = df.groupBy("remark_is_boilerplate").count()
print("\n  Boilerplate distribution:")
boilerplate_stats.show()

# Step 5: Normalize ward_name using geo_utils
print("\nStep 5: Normalizing ward names...")

# Create UDF for normalize_ward_name
normalize_ward_udf = F.udf(normalize_ward_name, F.StringType())

df = df.withColumn(
    "ward_name_normalized",
    normalize_ward_udf(F.col("Ward_Name"))
)

print("  ✓ Normalized ward names (lowercase, removed 'ward' suffix)")

# Show sample of normalized wards
print("\n  Sample ward normalizations:")
df.select("Ward_Name", "ward_name_normalized") \
  .distinct() \
  .orderBy("Ward_Name") \
  .show(10, truncate=False)

# Step 6: Data Quality Checks (DLT-style expectations)
print("\nStep 6: Running data quality checks...")

# Check for null critical fields
null_complaint_id = df.filter(F.col("Complaint_ID").isNull()).count()
null_grievance_date = df.filter(F.col("grievance_timestamp").isNull()).count()
null_ward = df.filter(F.col("Ward_Name").isNull()).count()

print(f"  Null Complaint_ID: {null_complaint_id:,}")
print(f"  Null grievance_timestamp: {null_grievance_date:,}")
print(f"  Null Ward_Name: {null_ward:,}")

if null_complaint_id > 0:
    print(f"  ⚠️ Warning: {null_complaint_id:,} rows with null Complaint_ID")
if null_grievance_date > 0:
    print(f"  ⚠️ Warning: {null_grievance_date:,} rows with null grievance_timestamp")

# Filter out rows with null critical fields for silver
print("\n  Filtering rows with valid Complaint_ID and grievance_timestamp...")
df_clean = df.filter(
    F.col("Complaint_ID").isNotNull() & 
    F.col("grievance_timestamp").isNotNull()
)

rows_filtered = bronze_df.count() - df_clean.count()
print(f"  Filtered {rows_filtered:,} invalid rows")
print(f"  Clean rows: {df_clean.count():,}")

# Step 7: Select and rename final columns
print("\nStep 7: Selecting final schema...")

silver_df = df_clean.select(
    F.col("_id").alias("id"),
    F.col("Complaint_ID").alias("complaint_id"),
    F.col("Category").alias("category"),
    F.col("Sub_Category").alias("sub_category"),
    F.col("grievance_timestamp"),
    F.col("grievance_date"),
    F.col("grievance_year"),
    F.col("grievance_month"),
    F.col("grievance_day_of_week"),
    F.col("Ward_Name").alias("ward_name"),
    F.col("ward_name_normalized"),
    F.col("Grievance_Status").alias("status"),
    F.col("outcome_label"),
    F.col("Staff_Remarks").alias("staff_remarks"),
    F.col("remark_length"),
    F.col("remark_is_boilerplate"),
    F.col("staff_name_cleaned").alias("staff_name"),
    F.col("staff_dept"),
    F.col("source_year")
)

print(f"  Final schema: {len(silver_df.columns)} columns")
print("\n  Final columns:")
for col in silver_df.columns:
    print(f"    - {col}")

# Step 8: Write to Delta table
print(f"\n=== Writing to {SILVER_TABLE} ===")

# Optimize write with partitioning by year
silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.autoOptimize.optimizeWrite", "true") \
    .option("delta.autoOptimize.autoCompact", "true") \
    .partitionBy("source_year") \
    .saveAsTable(SILVER_TABLE)

print(f"✓ Successfully wrote {df_clean.count():,} rows to {SILVER_TABLE}")

# Final validation
print("\n=== Silver Table Summary ===")
silver_table = spark.table(SILVER_TABLE)
print(f"Total rows: {silver_table.count():,}")
print(f"Total columns: {len(silver_table.columns)}")

print("\nOutcome label distribution:")
silver_table.groupBy("outcome_label").count().orderBy("outcome_label").show()

print("\nRows by year:")
silver_table.groupBy("source_year").count().orderBy("source_year").show()

print("\n✓ Silver layer transformation complete!")

# COMMAND ----------

# DBTITLE 1,Verify Silver Table Quality
from pyspark.sql import functions as F

# Quick verification of silver table
silver = spark.table("civic_lens.silver.bbmp_complaints_clean")

print("=== Silver Table Quality Report ===")
print(f"\nTotal Records: {silver.count():,}")
print(f"Schema: {len(silver.columns)} columns\n")

# Outcome label breakdown
print("Outcome Label Distribution:")
outcome_df = silver.groupBy("outcome_label") \
    .agg(
        F.count("*").alias("count"),
        (F.count("*") * 100.0 / silver.count()).alias("percentage")
    ) \
    .orderBy("outcome_label")
outcome_df.show()

print("Legend: 0=Resolved, 1=Closed, 2=Rejected, NULL=In-Progress\n")

# Top categories
print("Top 10 Complaint Categories:")
silver.groupBy("category") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(10, truncate=False)

# Department distribution
print("Top 10 Staff Departments:")
silver.groupBy("staff_dept") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(10, truncate=False)

# Boilerplate analysis
print("Remark Quality:")
silver.groupBy("remark_is_boilerplate") \
    .agg(
        F.count("*").alias("count"),
        (F.count("*") * 100.0 / silver.count()).alias("percentage")
    ) \
    .show()

# Sample of clean records
print("Sample Silver Records:")
display(
    silver.select(
        "complaint_id",
        "category",
        "grievance_date",
        "ward_name_normalized",
        "outcome_label",
        "staff_dept",
        "remark_length"
    ).limit(10)
)

# COMMAND ----------

# DBTITLE 1,Notebook Summary
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Silver Layer Transformation Complete
# MAGIC
# MAGIC ### Output Summary
# MAGIC * **Table Created:** `civic_lens.silver.bbmp_complaints_clean`
# MAGIC * **Total Records:** 766,648 complaints
# MAGIC * **Schema:** 19 enriched columns
# MAGIC * **Partitioning:** By source_year for optimal query performance
# MAGIC
# MAGIC ### Data Quality Metrics
# MAGIC * **Valid Records:** 100% (all rows have complaint_id and timestamp)
# MAGIC * **Boilerplate Remarks:** 88.4% (automated/generic responses)
# MAGIC * **Non-Boilerplate:** 11.6% (detailed human responses)
# MAGIC
# MAGIC ### Feature Engineering Results
# MAGIC
# MAGIC **Outcome Labels:**
# MAGIC * **Class 0 (Resolved):** 6,953 (0.9%)
# MAGIC * **Class 1 (Closed):** 702,410 (91.6%)
# MAGIC * **Class 2 (Rejected):** 32,853 (4.3%)
# MAGIC * **NULL (In-Progress):** 24,432 (3.2%)
# MAGIC
# MAGIC **Top Categories:**
# MAGIC 1. Electrical: 310,128
# MAGIC 2. Solid Waste: 195,153
# MAGIC 3. Road Maintenance: 111,535
# MAGIC 4. Forest: 34,618
# MAGIC 5. Health Dept: 29,924
# MAGIC
# MAGIC **Top Staff Departments:**
# MAGIC 1. AEE: 471,629
# MAGIC 2. JHI: 54,750
# MAGIC 3. AE: 51,422
# MAGIC 4. Customer Support: 37,850
# MAGIC
# MAGIC ### Schema Enhancements
# MAGIC * **Temporal:** `grievance_date`, `grievance_year`, `grievance_month`, `day_of_week`
# MAGIC * **Classification:** `outcome_label` (3-class)
# MAGIC * **Staff:** `staff_name`, `staff_dept`
# MAGIC * **Remark Quality:** `remark_length`, `remark_is_boilerplate`
# MAGIC * **Geography:** `ward_name_normalized`
# MAGIC
# MAGIC ### Next Steps
# MAGIC ➡️ **Notebook 03:** Aggregate metrics at ward and category levels for analytics
# MAGIC
# MAGIC ---
