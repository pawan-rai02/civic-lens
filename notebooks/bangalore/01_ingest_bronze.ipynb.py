# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Notebook Header
# MAGIC %md
# MAGIC # 01 - Bronze Layer: BBMP Complaint Data Ingestion
# MAGIC
# MAGIC **Pipeline Stage:** Bronze (Raw Data Lake)
# MAGIC
# MAGIC **Objective:** Ingest raw BBMP complaint JSON files from S3 (2020–2025) and create a unified Delta table.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Input
# MAGIC * **Source:** `s3://civiclens-data/bangalore/bbmp-{year}.json` (2020–2025)
# MAGIC * **Format:** Custom JSON structure with `{fields: [...], records: [...]}`
# MAGIC * **Data:** BBMP civic complaints from Bangalore's public grievance system
# MAGIC
# MAGIC ## Output
# MAGIC * **Table:** `civic_lens.bronze.bbmp_complaints_raw`
# MAGIC * **Format:** Delta Lake, partitioned by `source_year`
# MAGIC * **Records:** ~766K complaints across 6 years
# MAGIC * **Columns:** 10 (9 original fields + source_year)
# MAGIC
# MAGIC ## Key Operations
# MAGIC 1. Parse custom JSON format using multiLine mode
# MAGIC 2. Extract field metadata and sanitize column names
# MAGIC 3. Union DataFrames across years with schema drift handling
# MAGIC 4. Write optimized Delta table with partitioning
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Check JSON file format
# Let's first check what the JSON looks like
import pyspark.sql.functions as F

test_file = "s3://civiclens-data/bangalore/bbmp-2020.json"

# Try reading as text first to see the structure
print("=== Checking raw JSON structure ===")
raw_text = spark.read.text(test_file)
print(f"Total lines: {raw_text.count()}")
print("\nFirst 10 lines:")
display(raw_text.limit(10))

# COMMAND ----------

# DBTITLE 1,Parse Custom JSON Format and Write to Delta
from pyspark.sql import functions as F

# Configuration
S3_BASE_PATH = "s3://civiclens-data/bangalore/"
OUTPUT_TABLE = "civic_lens.bronze.bbmp_complaints_raw"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

print("=== Parsing Custom JSON Format ===")
print("These JSON files use {fields: [...], records: [...]} format\n")

unioned_dfs = []

for year in YEARS:
    file_path = f"{S3_BASE_PATH}bbmp-{year}.json"
    print(f"Processing {year}...")
    
    # Read as multiLine JSON since it's a single JSON object per file
    raw_df = spark.read.option("multiLine", "true").json(file_path)
    
    # Structure: {fields: [...], records: [...]}
    # Extract records array and explode it
    records_df = raw_df.select(F.explode("records").alias("record"))
    
    # Get field metadata to extract column names
    fields_array = raw_df.select("fields").first()[0]
    column_names = [field["id"] for field in fields_array]
    
    # Sanitize column names: replace spaces with underscores, remove special chars
    def sanitize_column_name(col_name):
        return col_name.replace(" ", "_").replace(",", "").replace(";", "").replace("{", "").replace("}", "").replace("(", "").replace(")", "").replace("\n", "").replace("\t", "").replace("=", "")
    
    sanitized_names = [sanitize_column_name(name) for name in column_names]
    
    # Expand the record array into separate columns
    expanded_df = records_df.select(
        *[F.col("record").getItem(i).alias(sanitized_names[i]) 
          for i in range(len(column_names))]
    )
    
    # Add source_year tag
    final_df = expanded_df.withColumn("source_year", F.lit(year))
    
    row_count = final_df.count()
    unioned_dfs.append(final_df)
    print(f"  ✓ {year}: {row_count:,} rows, {len(column_names)} columns")

# Union all years with allowMissingColumns for schema drift
print("\nUnioning all years...")
bronze_df = unioned_dfs[0]
for df in unioned_dfs[1:]:
    bronze_df = bronze_df.unionByName(df, allowMissingColumns=True)

print("✓ Union complete")

# Write to Delta
print("\n=== Writing to Delta Table ===")
print("Repartitioning...")
optimized_df = bronze_df.repartition(30, "source_year")

print(f"Writing to {OUTPUT_TABLE}...")
optimized_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.autoOptimize.optimizeWrite", "true") \
    .option("delta.autoOptimize.autoCompact", "true") \
    .saveAsTable(OUTPUT_TABLE)

print(f"✓ Successfully wrote to {OUTPUT_TABLE}")

# Summary
print("\n=== Bronze Table Summary ===")
final_table = spark.table(OUTPUT_TABLE)
total_rows = final_table.count()
print(f"Total rows: {total_rows:,}")
print(f"Total columns: {len(final_table.columns)}")

print("\nRow distribution by year:")
year_summary = final_table.groupBy("source_year") \
    .agg(F.count("*").alias("row_count")) \
    .orderBy("source_year")

display(year_summary)

# COMMAND ----------

# DBTITLE 1,Notebook Summary
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Bronze Layer Ingestion Complete
# MAGIC
# MAGIC ### Output Summary
# MAGIC * **Table Created:** `civic_lens.bronze.bbmp_complaints_raw`
# MAGIC * **Total Records:** 766,648 complaints
# MAGIC * **Year Range:** 2020–2025
# MAGIC * **Columns:** 10 (ID, Complaint_ID, Category, Sub_Category, Grievance_Date, Ward_Name, Grievance_Status, Staff_Remarks, Staff_Name, source_year)
# MAGIC
# MAGIC ### Data Quality Metrics
# MAGIC * **Schema Consistency:** All years unified with `allowMissingColumns`
# MAGIC * **Partitioning:** By source_year for query performance
# MAGIC * **Optimization:** Auto-optimize enabled (optimizeWrite, autoCompact)
# MAGIC
# MAGIC ### Year Distribution
# MAGIC | Year | Row Count |
# MAGIC |------|-----------|
# MAGIC | 2020 | 91,620    |
# MAGIC | 2021 | 103,504   |
# MAGIC | 2022 | 118,394   |
# MAGIC | 2023 | 119,140   |
# MAGIC | 2024 | 207,016   |
# MAGIC | 2025 | 126,974   |
# MAGIC
# MAGIC ### Next Steps
# MAGIC ➡️ **Notebook 02:** Clean and transform bronze data into silver layer with data quality checks
# MAGIC
# MAGIC ---
