# NYC 311 Complaints Data Pipeline

This folder contains the end-to-end data pipeline for processing and analyzing NYC 311 service request data.

## 📊 Pipeline Overview

The pipeline transforms raw NYC 311 complaint data through Bronze → Silver → Feature Engineering layers using Databricks Delta Lake.

```
Raw CSV Data (S3)
    ↓
[01_ingest_bronze] → Bronze Layer (civic_lens.bronze.nyc_311_raw)
    ↓
[02_clean_silver] → Silver Layer (civic_lens.silver.nyc_311_cleaned)
    ↓
[03_aggregate_silver] → Feature Table (civic_lens.silver.nyc_borough_agency_agg)
```

---

## 📁 Notebooks

### 1. `01_ingest_bronze.ipynb` - Data Ingestion

**Purpose**: Ingest raw NYC 311 complaint CSV data from S3 into Bronze layer.

**Input**: 
- Raw CSV file: `s3://civiclens-data/nyc311/nyc311_2020_2023.csv`
- 5.4M complaint records from 2022-2023

**Output**: 
- Table: `civic_lens.bronze.nyc_311_raw`
- Format: Delta Lake (parquet)

**What it does**:
- Reads CSV with schema inference
- Adds metadata columns: `ingestion_timestamp`, `source_file`
- Writes to Bronze layer with minimal transformation
- Preserves raw data for auditing and re-processing

---

### 2. `02_clean_silver.ipynb` - Data Cleaning & Transformation

**Purpose**: Clean, standardize, and enrich Bronze data into analysis-ready Silver layer.

**Input**: 
- Table: `civic_lens.bronze.nyc_311_raw` (5.4M records)

**Output**: 
- Table: `civic_lens.silver.nyc_311_cleaned` (4.97M records after filtering)

**What it does**:

1. **Parse & Standardize Dates**
   - Convert `created_date` and `closed_date` to proper timestamps
   - Extract temporal features: `dow_filed`, `hour_filed`, `month_filed`

2. **Calculate Derived Metrics**
   - `is_open`: Boolean flag for open complaints
   - `resolution_days`: Days between creation and closure
   - `never_resolved`: Flag for complaints open >90 days

3. **Clean Text Fields**
   - Use custom UDF `clean_text()` to normalize descriptor + resolution text
   - Remove punctuation, lowercase, combine fields

4. **Standardize Location Data**
   - Use custom UDF `normalize_borough()` for consistent borough names
   - Filter for valid NYC coordinates (latitude/longitude bounds)

5. **Data Quality Filtering**
   - Remove records with null critical fields
   - Remove invalid coordinates
   - **Result**: 4.97M clean records (432K removed)

**Quality Summary**:
- Total records: 4,967,746
- Open complaints: 55,757 (1.1%)
- Never resolved (>90 days): 55,757
- Average resolution time: 25.24 days
- Zero null boroughs after cleaning

---

### 3. `03_aggregate_silver.ipynb` - Rolling Feature Engineering

**Purpose**: Compute time-based rolling aggregate features for ML model training.

**Input**: 
- Table: `civic_lens.silver.nyc_311_cleaned` (4.97M records)

**Output**: 
- Table: `civic_lens.silver.nyc_borough_agency_agg` (4.49M feature records)

**Features Computed**:

1. **`borough_blackhole_rate`** (per borough)
   - Rolling 12-month rate of unresolved complaints
   - Measures: "How many complaints remain open?"
   - Window: 12 months trailing

2. **`agency_resolution_rate_hist`** (per agency)
   - Rolling 12-month resolution rate
   - Measures: "What % of complaints does this agency close?"
   - Window: 12 months trailing

3. **`agency_open_complaints_30d`** (per agency)
   - Count of open complaints in trailing 30 days
   - Measures: "Current open complaint volume"
   - Window: 30 days trailing

**Output Schema**:
```
borough                      STRING
agency                       STRING
created_date                 TIMESTAMP
borough_blackhole_rate       DOUBLE    (0.0 - 1.0)
agency_resolution_rate_hist  DOUBLE    (0.0 - 1.0)
agency_open_complaints_30d   LONG      (count)
```

---

## 🚀 Performance Optimization Story: From 2+ Hours to <5 Minutes

### The Problem

The original `03_aggregate_silver` notebook was taking **2+ hours** to compute rolling features because it was:
1. Running window functions on the full 5M row dataset
2. Using `rangeBetween()` with timestamp casting for every row
3. Performing 3 separate full-table scans with different windows
4. Then deduplicating 5M rows after computing features

### The Bug

On top of the performance issue, **all feature values were 0** because the code was checking:
- `status == "Open"` and `status == "Closed"` (title case)

But the actual data has:
- `status == "OPEN"` and `status == "CLOSED"` (uppercase)

### The Solution: Pre-Aggregation Strategy

Instead of computing rolling windows on 5M rows, we:
1. **Pre-aggregate to monthly/daily grain** (reducing data volume by 40-600x)
2. **Compute windows on aggregates** (much smaller dataset)
3. **Join back to original grain** (fast broadcast join)

This is a classic "aggregate first, then compute" pattern that outperforms any `repartition()` tuning.

---

## 📝 Code Comparison: Old vs Optimized

### Feature 1: Borough Blackhole Rate (12-month rolling)

#### ❌ OLD CODE (Slow + Buggy)

```python
from pyspark.sql import Window
from pyspark.sql import functions as F

nyc_clean = spark.table("civic_lens.silver.nyc_311_cleaned")

# Define 12-month window per borough
# Convert created_date to unix timestamp for numeric range window
boroughs_window_12m = Window.partitionBy("borough").orderBy(F.col("created_date").cast("long")).rangeBetween(
    -365 * 86400,  # 12 months in seconds (365 days)
    0
)

# Calculate rolling metrics per borough
# 🐛 BUG: "Open" should be "OPEN" (uppercase)
borough_agg = (
    nyc_clean  # ⚠️ PROBLEM: Operating on 5M rows!
    .withColumn(
        "total_complaints_12m",
        F.count("*").over(boroughs_window_12m)  # ⚠️ Expensive window on full table
    )
    .withColumn(
        "unresolved_complaints_12m",
        F.sum(F.when(F.col("status") == "Open", 1).otherwise(0)).over(boroughs_window_12m)  # 🐛 Wrong case
    )
    .withColumn(
        "borough_blackhole_rate",
        F.when(
            F.col("total_complaints_12m") > 0,
            F.col("unresolved_complaints_12m") / F.col("total_complaints_12m")
        ).otherwise(0.0)
    )
    .select("borough", "created_date", "borough_blackhole_rate")
    .distinct()  # ⚠️ Expensive deduplication of 5M rows
)

# Result: 40+ minutes runtime, all values = 0
```

**Problems**:
- ⚠️ Window function on 5M rows (expensive shuffle)
- ⚠️ `rangeBetween()` with timestamp casting for every row
- ⚠️ Deduplication after window (another shuffle)
- 🐛 Wrong case sensitivity (all results = 0)

---

#### ✅ OPTIMIZED CODE (Fast + Fixed)

```python
from pyspark.sql import Window
from pyspark.sql import functions as F

nyc_clean = spark.table("civic_lens.silver.nyc_311_cleaned")
print(f"Source records: {nyc_clean.count():,}")  # 4,967,746

# ✅ OPTIMIZATION: Pre-aggregate to MONTHLY borough level
# This reduces data volume by ~40,000x (from 5M rows to 126 monthly aggregates)
monthly_borough = (
    nyc_clean
    .withColumn("month", F.trunc("created_date", "month"))
    .groupBy("borough", "month")
    .agg(
        F.count("*").alias("total_complaints"),
        # ✅ BUG FIX: Use "OPEN" (uppercase)
        F.sum(F.when(F.col("status") == "OPEN", 1).otherwise(0)).alias("open_complaints")
    )
)

print(f"Monthly aggregates: {monthly_borough.count():,}")  # 126 rows!

# ✅ Apply 12-month rolling window on aggregated data (much faster!)
# Using rowsBetween instead of rangeBetween (simpler, faster)
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

# ✅ Join back to daily grain for final table
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

# Result: <2 minutes runtime, correct non-zero values
```

**Improvements**:
- ✅ Pre-aggregate 5M rows → 126 monthly aggregates (40,000x reduction!)
- ✅ Window on 126 rows instead of 5M (massive speedup)
- ✅ Simple `rowsBetween(-11, 0)` instead of complex `rangeBetween`
- ✅ Join back to daily grain (fast broadcast join)
- ✅ Fixed case sensitivity bug

---

### Feature 2: Agency Resolution Rate (12-month rolling)

#### ❌ OLD CODE

```python
# 🐛 BUG: "Closed" should be "CLOSED"
agency_window_12m = Window.partitionBy("agency").orderBy(F.col("created_date").cast("long")).rangeBetween(
    -365 * 86400,
    0
)

agency_resolution = (
    nyc_clean  # ⚠️ 5M rows
    .withColumn(
        "total_complaints_12m",
        F.count("*").over(agency_window_12m)
    )
    .withColumn(
        "resolved_complaints_12m",
        F.sum(
            F.when(
                (F.col("status") == "Closed") & (F.col("closed_date").isNotNull()),  # 🐛 Wrong case
                1
            ).otherwise(0)
        ).over(agency_window_12m)
    )
    .withColumn(
        "agency_resolution_rate_hist",
        F.when(
            F.col("total_complaints_12m") > 0,
            F.col("resolved_complaints_12m") / F.col("total_complaints_12m")
        ).otherwise(0.0)
    )
    .select("agency", "created_date", "agency_resolution_rate_hist")
    .distinct()
)

# Result: 40+ minutes runtime, all values = 0
```

---

#### ✅ OPTIMIZED CODE

```python
# ✅ Pre-aggregate to MONTHLY agency level
monthly_agency = (
    nyc_clean
    .withColumn("month", F.trunc("created_date", "month"))
    .groupBy("agency", "month")
    .agg(
        F.count("*").alias("total_complaints"),
        # ✅ BUG FIX: Use "CLOSED" (uppercase)
        F.sum(
            F.when(
                (F.col("status") == "CLOSED") & (F.col("closed_date").isNotNull()),
                1
            ).otherwise(0)
        ).alias("resolved_complaints")
    )
)

print(f"Monthly agency aggregates: {monthly_agency.count():,}")  # 297 rows

# ✅ Apply window on aggregates
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

# ✅ Join back to daily grain
agency_resolution = (
    nyc_clean
    .withColumn("month", F.trunc("created_date", "month"))
    .select("agency", "created_date", "month")
    .distinct()
    .join(monthly_agency_metrics, on=["agency", "month"], how="left")
    .select("agency", "created_date", "agency_resolution_rate_hist")
    .fillna({"agency_resolution_rate_hist": 0.0})
)

# Result: <2 minutes runtime, correct values (e.g. 1.0 = 100% resolved)
```

**Improvements**:
- ✅ 5M rows → 297 monthly agency aggregates (16,700x reduction)
- ✅ Fixed case sensitivity bug

---

### Feature 3: Agency Open Complaints (30-day rolling)

#### ❌ OLD CODE

```python
agency_window_30d = Window.partitionBy("agency").orderBy(F.col("created_date").cast("long")).rangeBetween(
    -30 * 86400,
    0
)

agency_open = (
    nyc_clean  # ⚠️ 5M rows
    .withColumn(
        "agency_open_complaints_30d",
        F.sum(
            F.when(F.col("status") == "Open", 1).otherwise(0)  # 🐛 Wrong case
        ).over(agency_window_30d)
    )
    .select("agency", "created_date", "agency_open_complaints_30d")
    .distinct()
)

# Result: 40+ minutes runtime, all values = 0
```

---

#### ✅ OPTIMIZED CODE

```python
# ✅ Pre-aggregate to DAILY agency level (appropriate for 30-day window)
daily_agency = (
    nyc_clean
    .withColumn("date", F.to_date("created_date"))
    .groupBy("agency", "date")
    .agg(
        # ✅ BUG FIX: Use "OPEN" (uppercase)
        F.sum(F.when(F.col("status") == "OPEN", 1).otherwise(0)).alias("open_complaints")
    )
)

print(f"Daily agency aggregates: {daily_agency.count():,}")  # 7,747 rows

# ✅ Apply 30-day rolling window on aggregates
agency_window_30d = Window.partitionBy("agency").orderBy("date").rowsBetween(-29, 0)

daily_agency_metrics = (
    daily_agency
    .withColumn(
        "agency_open_complaints_30d",
        F.sum("open_complaints").over(agency_window_30d)
    )
    .select("agency", "date", "agency_open_complaints_30d")
)

# ✅ Join back to original grain
agency_open = (
    nyc_clean
    .withColumn("date", F.to_date("created_date"))
    .select("agency", "created_date", "date")
    .distinct()
    .join(daily_agency_metrics, on=["agency", "date"], how="left")
    .select("agency", "created_date", "agency_open_complaints_30d")
    .fillna({"agency_open_complaints_30d": 0})
)

# Result: <1 minute runtime, correct values (e.g. 204 open complaints)
```

**Improvements**:
- ✅ 5M rows → 7,747 daily aggregates (640x reduction)
- ✅ Daily grain is appropriate for 30-day window
- ✅ Fixed case sensitivity bug

---

## 📈 Performance Results

### Before Optimization

| Feature | Rows Processed | Window Type | Runtime | Output |
|---------|----------------|-------------|---------|--------|
| Borough blackhole rate | 5M | rangeBetween (timestamp) | ~40 min | ❌ All zeros |
| Agency resolution rate | 5M | rangeBetween (timestamp) | ~40 min | ❌ All zeros |
| Agency open complaints | 5M | rangeBetween (timestamp) | ~40 min | ❌ All zeros |
| **TOTAL** | - | - | **~2 hours** | **❌ Buggy** |

### After Optimization

| Feature | Aggregates | Window Type | Runtime | Output |
|---------|------------|-------------|---------|--------|
| Borough blackhole rate | 126 monthly | rowsBetween (simple) | <2 min | ✅ Correct values |
| Agency resolution rate | 297 monthly | rowsBetween (simple) | <2 min | ✅ Correct values |
| Agency open complaints | 7,747 daily | rowsBetween (simple) | <1 min | ✅ Correct values |
| **TOTAL** | - | - | **<5 min** | **✅ Fixed** |

### Key Improvements

1. **24x faster execution** (120 min → 5 min)
2. **40,000x data reduction** for borough metrics (5M → 126)
3. **16,700x data reduction** for agency resolution (5M → 297)
4. **640x data reduction** for 30-day metrics (5M → 7,747)
5. **Fixed bug**: All values now non-zero and correct
6. **Simpler code**: `rowsBetween` instead of complex `rangeBetween` with casting
7. **Better testing**: Smaller aggregates easier to validate

---

## 🎯 Key Takeaways

### The Optimization Pattern: "Aggregate First, Then Compute"

When working with time-series rolling windows:

1. **DON'T**: Run window functions on raw transaction-level data
2. **DO**: Pre-aggregate to the appropriate grain first
   - For 12-month windows → monthly aggregates
   - For 30-day windows → daily aggregates
3. **THEN**: Compute windows on aggregates (10-1000x smaller)
4. **FINALLY**: Join back to original grain if needed

This pattern:
- ✅ Reduces shuffle volume by orders of magnitude
- ✅ Makes windows operations fast (fewer rows)
- ✅ Simplifies window logic (`rowsBetween` vs `rangeBetween`)
- ✅ Easier to test and validate (inspect aggregates directly)

### Case Sensitivity Matters

Always check the actual data values before writing filters:
```python
# ❌ WRONG (assumes title case)
F.when(F.col("status") == "Open", 1)

# ✅ CORRECT (check actual data first)
F.when(F.col("status") == "OPEN", 1)
```

Use `df.select("status").distinct().show()` to verify actual values.

---

## 🔧 Usage

### Running the Pipeline

1. **Bronze Ingestion**:
   ```python
   %run ./01_ingest_bronze
   ```

2. **Silver Cleaning**:
   ```python
   %run ./02_clean_silver
   ```

3. **Feature Engineering** (optimized):
   ```python
   %run ./03_aggregate_silver
   ```

### Output Tables

```python
# Bronze layer
bronze = spark.table("civic_lens.bronze.nyc_311_raw")

# Silver layer
silver = spark.table("civic_lens.silver.nyc_311_cleaned")

# Feature table
features = spark.table("civic_lens.silver.nyc_borough_agency_agg")
```

---

## 📊 Sample Output

### Feature Table Sample

```
+-------------+-------+-------------------+----------------------+---------------------------+--------------------------+
|      borough| agency|       created_date|borough_blackhole_rate|agency_resolution_rate_hist|agency_open_complaints_30d|
+-------------+-------+-------------------+----------------------+---------------------------+--------------------------+
|    MANHATTAN|   NYPD|2022-04-02 16:03:58|           0.001485819|                       0.99|                        15|
|     BROOKLYN|    DOB|2022-03-15 08:30:22|           0.002134567|                       0.87|                       204|
|       QUEENS|    DEP|2022-02-28 14:22:10|           0.001823456|                       0.95|                        42|
+-------------+-------+-------------------+----------------------+---------------------------+--------------------------+
```

### Average Values

```
avg_blackhole_rate: 0.00148
avg_resolution_rate: 0.946
avg_open_complaints_30d: 12.5
```

---

## 📚 References

- [Databricks Window Functions](https://docs.databricks.com/sql/language-manual/functions/window-functions.html)
- [NYC 311 Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)
- [PySpark Window Functions Guide](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/window.html)

---

## 📝 Notes

- All timestamps are in UTC
- Borough names are normalized to uppercase
- Rates are expressed as 0.0-1.0 (not percentages)
- Window functions use trailing periods (look-back only)
- Feature table is at daily grain (one row per borough/agency/date combination)

---

**Last Updated**: 2026-06-18  
**Databricks Runtime**: Serverless (CPU)  
**Author**: Civic Lens Team