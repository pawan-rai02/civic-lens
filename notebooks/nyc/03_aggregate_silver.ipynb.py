# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Aggregate Silver - NYC Historical Rolling Features
# MAGIC %md
# MAGIC # NYC Aggregate Silver: Historical Rolling Features
# MAGIC
# MAGIC **Purpose**: Compute historical rolling features using group-by-window logic for NYC complaints.
# MAGIC
# MAGIC **Features**:
# MAGIC * `borough_blackhole_rate` — Rolling 12-month rate of unresolved complaints per borough
# MAGIC * `agency_resolution_rate_hist` — Rolling 12-month resolution rate per agency
# MAGIC * `agency_open_complaints_30d` — Count of open complaints in trailing 30 days per agency
# MAGIC
# MAGIC **Input**: `civic_lens.silver.nyc_311_cleaned`
# MAGIC
# MAGIC **Output**: `civic_lens.silver.nyc_borough_agency_agg` (aggregate table to be joined back in feature engineering)
# MAGIC
# MAGIC **Method**: PySpark `groupBy` + `Window` functions over trailing 12-month and 30-day periods

# COMMAND ----------

# DBTITLE 1,Borough Blackhole Rate (12-month rolling)
from pyspark.sql import Window
from pyspark.sql import functions as F

# Load clean data
nyc_clean = spark.table("civic_lens.silver.nyc_311_cleaned")

print(f"Source records: {nyc_clean.count():,}")

# OPTIMIZATION: Pre-aggregate to MONTHLY borough level before window functions
# This reduces data volume by ~150x (from 5M rows to ~30K monthly aggregates)
monthly_borough = (
    nyc_clean
    .withColumn("month", F.trunc("created_date", "month"))
    .groupBy("borough", "month")
    .agg(
        F.count("*").alias("total_complaints"),
        # BUG FIX: Use "OPEN" (uppercase) not "Open"
        F.sum(F.when(F.col("status") == "OPEN", 1).otherwise(0)).alias("open_complaints")
    )
)

print(f"Monthly aggregates: {monthly_borough.count():,} (reduced from {nyc_clean.count():,})")

# Apply 12-month rolling window on aggregated data (much faster!)
borough_window_12m = Window.partitionBy("borough").orderBy("month").rowsBetween(-11, 0)

monthly_borough_metrics = (
    monthly_borough
    .withColumn("total_12m", F.sum("total_complaints").over(borough_window_12m))
    .withColumn("open_12m", F.sum("open_complaints").over(borough_window_12m))
    .withColumn(
        "borough_blackhole_rate",
        F.when(F.col("total_12m") > 0, F.col("open_12m") / F.col("total_12m")).otherwise(0.0)
    )
    .select("borough", "month", "borough_blackhole_rate")
)

# Join back to daily grain for final table
borough_agg = (
    nyc_clean
    .withColumn("month", F.trunc("created_date", "month"))
    .select("borough", "created_date", "month")
    .distinct()
    .join(monthly_borough_metrics, on=["borough", "month"], how="left")
    .select("borough", "created_date", "borough_blackhole_rate")
    .fillna({"borough_blackhole_rate": 0.0})
)

print("\nBorough blackhole rate (sample):")
display(borough_agg.orderBy(F.desc("borough_blackhole_rate")).limit(10))

# COMMAND ----------

# DBTITLE 1,Agency Resolution Rate Historical (12-month rolling)
# OPTIMIZATION: Pre-aggregate to MONTHLY agency level before window functions
monthly_agency = (
    nyc_clean
    .withColumn("month", F.trunc("created_date", "month"))
    .groupBy("agency", "month")
    .agg(
        F.count("*").alias("total_complaints"),
        # BUG FIX: Use "CLOSED" (uppercase) not "Closed"
        F.sum(
            F.when(
                (F.col("status") == "CLOSED") & (F.col("closed_date").isNotNull()),
                1
            ).otherwise(0)
        ).alias("resolved_complaints")
    )
)

print(f"Monthly agency aggregates: {monthly_agency.count():,}")

# Apply 12-month rolling window on aggregated data
agency_window_12m = Window.partitionBy("agency").orderBy("month").rowsBetween(-11, 0)

monthly_agency_metrics = (
    monthly_agency
    .withColumn("total_12m", F.sum("total_complaints").over(agency_window_12m))
    .withColumn("resolved_12m", F.sum("resolved_complaints").over(agency_window_12m))
    .withColumn(
        "agency_resolution_rate_hist",
        F.when(F.col("total_12m") > 0, F.col("resolved_12m") / F.col("total_12m")).otherwise(0.0)
    )
    .select("agency", "month", "agency_resolution_rate_hist")
)

# Join back to daily grain
agency_resolution = (
    nyc_clean
    .withColumn("month", F.trunc("created_date", "month"))
    .select("agency", "created_date", "month")
    .distinct()
    .join(monthly_agency_metrics, on=["agency", "month"], how="left")
    .select("agency", "created_date", "agency_resolution_rate_hist")
    .fillna({"agency_resolution_rate_hist": 0.0})
)

print("\nAgency resolution rate historical (sample):")
display(agency_resolution.orderBy(F.desc("agency_resolution_rate_hist")).limit(10))

# COMMAND ----------

# DBTITLE 1,Agency Open Complaints (30-day rolling)
# OPTIMIZATION: Pre-aggregate to DAILY agency level for 30-day window
# (Daily grain is appropriate for 30-day window)
daily_agency = (
    nyc_clean
    .withColumn("date", F.to_date("created_date"))
    .groupBy("agency", "date")
    .agg(
        # BUG FIX: Use "OPEN" (uppercase) not "Open"
        F.sum(F.when(F.col("status") == "OPEN", 1).otherwise(0)).alias("open_complaints")
    )
)

print(f"Daily agency aggregates: {daily_agency.count():,}")

# Apply 30-day rolling window on aggregated data
agency_window_30d = Window.partitionBy("agency").orderBy("date").rowsBetween(-29, 0)

daily_agency_metrics = (
    daily_agency
    .withColumn(
        "agency_open_complaints_30d",
        F.sum("open_complaints").over(agency_window_30d)
    )
    .select("agency", "date", "agency_open_complaints_30d")
)

# Join back to original grain
agency_open = (
    nyc_clean
    .withColumn("date", F.to_date("created_date"))
    .select("agency", "created_date", "date")
    .distinct()
    .join(daily_agency_metrics, on=["agency", "date"], how="left")
    .select("agency", "created_date", "agency_open_complaints_30d")
    .fillna({"agency_open_complaints_30d": 0})
)

print("\nAgency open complaints 30-day (sample):")
display(agency_open.orderBy(F.desc("agency_open_complaints_30d")).limit(10))

# COMMAND ----------

# DBTITLE 1,Join All Aggregates and Save
# Join all three aggregate features together
# Join on borough + agency + created_date to create comprehensive aggregate table

# First, create a base with all unique borough/agency/date combinations from original data
base_keys = (
    nyc_clean
    .select("borough", "agency", "created_date")
    .distinct()
)

# Join all aggregate features
nyc_agg_final = (
    base_keys
    .join(borough_agg, on=["borough", "created_date"], how="left")
    .join(agency_resolution, on=["agency", "created_date"], how="left")
    .join(agency_open, on=["agency", "created_date"], how="left")
    # Fill nulls with 0 for new agencies/boroughs without history
    .fillna({
        "borough_blackhole_rate": 0.0,
        "agency_resolution_rate_hist": 0.0,
        "agency_open_complaints_30d": 0
    })
)

print("\nFinal aggregate table:")
print(f"Total records: {nyc_agg_final.count():,}")
nyc_agg_final.printSchema()
display(nyc_agg_final.limit(10))

# COMMAND ----------

# DBTITLE 1,Write to Silver Aggregate Table
# Write aggregate features to Delta table
output_table = "civic_lens.silver.nyc_borough_agency_agg"

nyc_agg_final.write.format("delta").mode("overwrite").saveAsTable(output_table)

print(f"\n✓ Successfully wrote aggregate features to {output_table}")
print(f"Total records: {spark.table(output_table).count():,}")

# Show summary statistics
print("\nAggregate feature summary:")
display(
    spark.table(output_table)
    .select(
        F.mean("borough_blackhole_rate").alias("avg_blackhole_rate"),
        F.mean("agency_resolution_rate_hist").alias("avg_resolution_rate"),
        F.mean("agency_open_complaints_30d").alias("avg_open_complaints_30d")
    )
)

# COMMAND ----------

# DBTITLE 1,Validation and Quality Checks
# Quality checks on aggregate table
agg_table = spark.table("civic_lens.silver.nyc_borough_agency_agg")

print("=== Aggregate Table Quality Checks ===")

# 1. Check for nulls
null_counts = agg_table.select(
    [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in agg_table.columns]
).collect()[0].asDict()
print("\n1. Null counts:")
for col, count in null_counts.items():
    if count > 0:
        print(f"   {col}: {count:,}")

# 2. Check value ranges (rates should be 0-1)
print("\n2. Rate value ranges (should be 0-1):")
rate_ranges = agg_table.select(
    F.min("borough_blackhole_rate").alias("min_blackhole"),
    F.max("borough_blackhole_rate").alias("max_blackhole"),
    F.min("agency_resolution_rate_hist").alias("min_resolution"),
    F.max("agency_resolution_rate_hist").alias("max_resolution")
).collect()[0]
for field in rate_ranges.asDict():
    print(f"   {field}: {rate_ranges[field]:.4f}")

# 3. Distribution by borough and agency
print("\n3. Records per borough:")
display(agg_table.groupBy("borough").count().orderBy(F.desc("count")))

print("\n4. Top agencies by record count:")
display(agg_table.groupBy("agency").count().orderBy(F.desc("count")).limit(10))

print("\n✓ Quality checks complete")

# COMMAND ----------


