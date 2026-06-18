# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,NYC 311 Silver Layer - Data Cleaning
# MAGIC %md
# MAGIC # NYC 311 Silver Layer - Data Cleaning
# MAGIC
# MAGIC This notebook cleans and transforms the bronze layer data:
# MAGIC - Parse and standardize date fields
# MAGIC - Calculate derived metrics (resolution_days, is_open, never_resolved)
# MAGIC - Extract temporal features (day of week, hour, month)
# MAGIC - Clean and combine text fields
# MAGIC - Standardize location data
# MAGIC
# MAGIC **Source**: `civic_lens.bronze.nyc_311_raw`
# MAGIC
# MAGIC **Target**: `civic_lens.silver.nyc_311_cleaned`

# COMMAND ----------

# DBTITLE 1,Import libraries
from pyspark.sql import functions as F
from pyspark.sql.types import *
import re

# COMMAND ----------

# DBTITLE 1,Set catalog and schema
catalog = "civic_lens"
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS silver")
spark.sql(f"USE SCHEMA silver")

print(f"Using: {catalog}.silver")

# COMMAND ----------

# DBTITLE 1,Read bronze data
df_bronze = spark.table("civic_lens.bronze.nyc_311_raw")
print(f"Bronze records: {df_bronze.count():,}")
print("\nBronze schema:")
df_bronze.printSchema()

# COMMAND ----------

# DBTITLE 1,Clean and transform data
df_silver = (
    df_bronze
    # complaint_id: string from unique_key
    .withColumn("complaint_id", F.col("unique_key").cast("string"))
    
    # created_date, closed_date: timestamp parsed
    .withColumn("created_date", F.to_timestamp("created_date"))
    .withColumn("closed_date", F.to_timestamp("closed_date"))
    
    # is_open: boolean - closed_date IS NULL
    .withColumn("is_open", F.col("closed_date").isNull())
    
    # resolution_days: double - datediff, null if open
    .withColumn(
        "resolution_days",
        F.when(
            F.col("is_open") == False,
            F.datediff(F.col("closed_date"), F.col("created_date"))
        ).otherwise(F.lit(None))
    )
    
    # never_resolved: int (0/1) - is_open AND age > 90d
    .withColumn(
        "never_resolved",
        F.when(
            F.col("is_open") & (F.datediff(F.current_date(), F.col("created_date")) > 90),
            1
        ).otherwise(0)
    )
    
    # dow_filed, hour_filed, month_filed: int from created_date
    .withColumn("dow_filed", F.dayofweek("created_date").cast("int"))
    .withColumn("hour_filed", F.hour("created_date").cast("int"))
    .withColumn("month_filed", F.month("created_date").cast("int"))
    
    # clean_text: string - cleaned descriptor + resolution_description
    .withColumn(
        "clean_text",
        F.concat_ws(
            " ",
            F.coalesce(F.trim(F.col("descriptor")), F.lit("")),
            F.coalesce(F.trim(F.col("resolution_description")), F.lit(""))
        )
    )
    
    # borough: string - direct, standardized casing
    .withColumn("borough", F.upper(F.trim(F.col("borough"))))
    
    # borough_normalized: string - normalized for geojson joins (lowercase, no UNSPECIFIED)
    .withColumn(
        "borough_normalized",
        F.when(F.col("borough") != "UNSPECIFIED", 
               F.lower(F.trim(F.col("borough"))))
        .otherwise(None)
    )
    
    # Clean location fields - use try_cast to handle malformed values
    .withColumn("latitude", F.expr("try_cast(latitude as double)"))
    .withColumn("longitude", F.expr("try_cast(longitude as double)"))
    
    # Standardize other text fields
    .withColumn("complaint_type", F.trim(F.col("complaint_type")))
    .withColumn("agency", F.upper(F.trim(F.col("agency"))))
    .withColumn("status", F.upper(F.trim(F.col("status"))))
    
    # Add processing timestamp
    .withColumn("processed_timestamp", F.current_timestamp())
)

print(f"Silver records (before filtering): {df_silver.count():,}")

# COMMAND ----------

# DBTITLE 1,Filter invalid records
# Keep only records with valid data
df_silver_filtered = (
    df_silver
    .filter(F.col("created_date").isNotNull())
    .filter(F.col("complaint_id").isNotNull())
    .filter(F.col("complaint_type").isNotNull())
    # Filter valid NYC coordinates (approximate bounds)
    # Note: latitude/longitude may contain NULL after try_cast handles malformed values
    .filter(F.col("latitude").isNotNull())
    .filter(F.col("longitude").isNotNull())
    .filter(
        (F.col("latitude").between(40.4, 41.0)) & 
        (F.col("longitude").between(-74.3, -73.7))
    )
)

print(f"Filtered silver records: {df_silver_filtered.count():,}")
print(f"Records removed: {df_silver.count() - df_silver_filtered.count():,}")

# COMMAND ----------

# DBTITLE 1,Preview cleaned data
# Display sample to verify transformations
print("Sample silver records:")
display(df_silver_filtered.select(
    "complaint_id", "created_date", "closed_date", "is_open", 
    "resolution_days", "never_resolved", "dow_filed", "hour_filed",
    "complaint_type", "borough", "clean_text"
).limit(10))

# COMMAND ----------

# DBTITLE 1,Data quality summary
# Data quality checks
print("Data Quality Summary:")
print(f"- Total records: {df_silver_filtered.count():,}")
print(f"- Open complaints: {df_silver_filtered.filter(F.col('is_open')).count():,}")
print(f"- Never resolved (>90 days): {df_silver_filtered.filter(F.col('never_resolved') == 1).count():,}")
print(f"- Null borough: {df_silver_filtered.filter(F.col('borough').isNull()).count()}")
print(f"- Avg resolution days (closed): {df_silver_filtered.filter(F.col('is_open') == False).agg(F.avg('resolution_days')).collect()[0][0]:.2f}")

print("\nBorough distribution:")
df_silver_filtered.groupBy("borough").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# DBTITLE 1,Write to silver table
target_table = "civic_lens.silver.nyc_311_cleaned"

(
    df_silver_filtered.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(target_table)
)

print(f"✓ Data written to {target_table}")

# COMMAND ----------

# DBTITLE 1,Verify silver table
df_verify = spark.table(target_table)
print(f"Table: {target_table}")
print(f"Record count: {df_verify.count():,}")
print(f"\nColumn list:")
for col in df_verify.columns:
    print(f"  - {col}")
print(f"\nSample data:")
display(df_verify.limit(5))

# COMMAND ----------


