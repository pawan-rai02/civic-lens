# NYC 311 Complaints Data Pipeline

> **End-to-end machine learning pipeline for analyzing and scoring NYC 311 service requests**

This folder contains a complete production-grade data pipeline that processes 5.4M NYC 311 complaint records through Bronze → Silver → Gold layers, trains ML models for resolution prediction, and generates risk scores for 55K+ open complaints across all NYC boroughs.

## 📊 Pipeline Overview

```
Raw CSV (S3: 5.4M records)
    ↓
[01] Bronze Ingestion → 5.4M raw records
    ↓
[02] Silver Cleaning → 4.97M cleaned records (91.2% retained)
    ↓
[03] Feature Engineering → 4.49M records + rolling temporal features (24x faster via pre-aggregation)
    ↓
[04] NLP Features → 896K records + 52 NLP features (TF-IDF, SVD, LDA)
    ↓
[05] Gold Layer → 896K records with 67 ML-ready features
    ↓
[06] Model Training → 2 production models (67.3% R², 97.96% AUC)
    ↓
[07] Borough Scoring → 401 risk scores for 55,757 open complaints
```

## 🗂️ Notebooks Quick Reference

| # | Notebook | Purpose | Input | Output | Key Metric |
|---|----------|---------|-------|--------|------------|
| 01 | `ingest_bronze` | Data ingestion from S3 | CSV (5.4M rows) | Bronze table | Raw data preserved |
| 02 | `clean_silver` | Data cleaning & quality | Bronze (5.4M) | Silver (4.97M) | 91.2% retention |
| 03 | `aggregate_silver` | Rolling feature engineering | Silver (4.97M) | Aggregates (4.49M) | **24x speedup** |
| 04 | `nlp_features` | NLP feature extraction | Silver (4.97M) | NLP features (896K) | 70.9% variance retained |
| 05 | `build_gold` | Join to Gold training table | 3 tables | Gold (896K × 67 cols) | ML-ready dataset |
| 06 | `train_models` | ML model training | Gold (896K) | 2 UC models | 67.3% R², 97.96% AUC |
| 07 | `score_boroughs` | Production scoring | 55K open complaints | 401 risk scores | 23.8% flagged MEDIUM |

---

## 📁 Detailed Notebook Documentation

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
   - **Result**: 4.97M clean records (432K removed, 91.2% retention)

**Quality Summary**:
- Total records: 4,967,746
- Open complaints: 55,757 (1.1%)
- Never resolved (>90 days): 55,757
- Average resolution time: 25.24 days
- Zero null boroughs after cleaning

---

### 3. `03_aggregate_silver.ipynb` - Rolling Feature Engineering ⚡

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

**Performance Achievement**: 
- ⚡ **24x speedup** (2+ hours → <5 minutes)
- ✅ Fixed critical case-sensitivity bug (all values were 0)
- 🎯 Pre-aggregation strategy: 40,000x data reduction for window operations

**Output Schema**:
```
borough                      STRING
agency                       STRING
created_date                 TIMESTAMP
borough_blackhole_rate       DOUBLE    (0.0 - 1.0)
agency_resolution_rate_hist  DOUBLE    (0.0 - 1.0)
agency_open_complaints_30d   LONG      (count)
```

> 💡 **See detailed optimization story below** for the complete before/after code comparison and performance breakdown.

---

### 4. `04_nlp_features.ipynb` - NLP Feature Engineering

**Purpose**: Extract NLP features from complaint text using TF-IDF, SVD, and topic modeling for ML model training.

**Input**: 
- Table: `civic_lens.silver.nyc_311_cleaned` (4.97M records with `clean_text` column)

**Output**: 
- Table: `civic_lens.silver.nyc_nlp_features` (896K records with 53 feature columns)

**What it does**:

1. **Urgency Score Extraction**
   - Count of urgency keywords (emergency, dangerous, urgent, fire, flooding, etc.)
   - 22 predefined keywords for high-priority situations
   - **Distribution**: 867K (96.8%) with score 0, 23K (2.6%) with score 1, 5K (0.6%) with score 2+

2. **TF-IDF + SVD Dimensionality Reduction**
   - TF-IDF vectorization with 5,000 max features
   - SVD reduction to 50 dimensions for efficiency
   - **Explained variance**: 70.9% retained in 50 dimensions
   - Min/max document frequency filtering (min_df=5, max_df=0.95)
   - Output: 50 features (`tfidf_feat_1` through `tfidf_feat_50`)

3. **LDA Topic Modeling**
   - Latent Dirichlet Allocation with 12 topics
   - Extracts dominant topic ID per complaint
   - **Top 5 topics** (by document count):
     * Topic 3: 135,514 docs (15.1%)
     * Topic 8: 117,619 docs (13.1%)
     * Topic 6: 102,585 docs (11.4%)
     * Topic 1: 99,281 docs (11.1%)
     * Topic 4: 87,811 docs (9.8%)

**Optimization Approach**:
- **Challenge**: UDF-based distributed processing caused OOM errors on 5M rows
- **Solution**: Sample-based processing on driver
  * Sample 900K records (18.1% of full dataset, 3x training data)
  * Process in batches on driver (avoids executor memory issues)
  * **Runtime**: ~10-15 minutes (vs. repeated OOM failures)
  * Maintains representative sample for downstream ML training

**Key Metrics**:
- Input: 4,967,746 records
- Output: 896,079 records (18.1% sample)
- Features generated: 52 (1 urgency + 1 topic + 50 TF-IDF)
- TF-IDF vocabulary: 5,000 terms
- Explained variance: 70.9%

---

## 🚀 Performance Optimization Story: From 2+ Hours to <5 Minutes