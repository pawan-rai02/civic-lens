# CivicLens: Multi-City Civic Complaints Analytics Platform

**A production-grade ML platform analyzing 6.2M+ civic complaints across two global cities**

This repository contains end-to-end data engineering and machine learning pipelines for analyzing civic complaint data from NYC (5.4M records) and Bangalore (766K records), implementing medallion architecture (Bronze → Silver → Gold) to predict outcomes and identify high-risk areas for municipal intervention.

---

## 🌍 Project Overview

### Business Problem
Municipal governments process millions of civic complaints annually. CivicLens builds predictive analytics systems to:
* **Predict complaint outcomes** with 97%+ accuracy
* **Identify high-risk geographic areas** requiring immediate intervention
* **Quantify service quality** across boroughs/wards and agencies/departments
* **Enable proactive resource allocation** through ML-powered risk scoring

### Platform Architecture
```
📁 S3 Raw Data (CSV/JSON)
    ↓
🥉 Bronze Layer (Raw Ingestion)
    ├─ Minimal transformation
    ├─ Schema validation
    └─ Audit trail preservation
    ↓
🥈 Silver Layer (Cleaned & Enriched)
    ├─ Data quality checks
    ├─ Feature engineering
    ├─ NLP processing (TF-IDF, SVD, LDA)
    └─ Aggregations
    ↓
🥇 Gold Layer (ML-Ready)
    ├─ Unified feature tables
    ├─ Zero nulls in critical features
    └─ Perfect join coverage
    ↓
🤖 ML Models (Trained & Deployed)
    ├─ XGBoost / Gradient Boosting
    ├─ Logistic Regression baselines
    └─ MLflow tracking
    ↓
📍 Risk Scoring (Operational Output)
    ├─ Batch inference
    ├─ Geographic risk aggregation
    └─ 3-tier classification (High/Medium/Low)
```

---

## 📂 Folder Structure

```
notebooks/
├── README.md                    # This file - Platform overview
├── nyc/                         # NYC 311 Service Requests Pipeline
│   ├── README.md               # NYC pipeline documentation
│   ├── 01_ingest_bronze.ipynb
│   ├── 02_clean_silver.ipynb
│   ├── 03_aggregate_silver.ipynb
│   ├── 04_nlp_features.ipynb
│   ├── 05_build_gold.ipynb
│   ├── 06_train_models.ipynb
│   └── 07_score_boroughs.ipynb
└── bangalore/                   # BBMP Bangalore Complaints Pipeline
    ├── README.md               # Bangalore pipeline documentation
    ├── 01_ingest_bronze.ipynb
    ├── 02_clean_silver.ipynb
    ├── 03_aggregate_silver.ipynb
    ├── 04_nlp_features.ipynb
    ├── 05_build_gold.ipynb
    ├── 06_train_models.ipynb
    └── 07_score_wards.ipynb
```

---

## 🗽 NYC Pipeline Summary

**Data Source:** NYC 311 Service Requests (2022-2023)  
**Scale:** 5.4M complaints → 4.97M cleaned records  
**Geography:** 5 boroughs (Manhattan, Brooklyn, Queens, Bronx, Staten Island)  
**Output:** 401 borough-agency risk scores for 55,757 open complaints

### Pipeline Stages (7 Notebooks)

| Stage | Notebook | Input | Output | Key Achievement |
|-------|----------|-------|--------|-----------------|
| **01** | `ingest_bronze` | CSV (5.4M rows) | Bronze table | Raw data preserved |
| **02** | `clean_silver` | Bronze (5.4M) | Silver (4.97M) | 91.2% retention |
| **03** | `aggregate_silver` | Silver (4.97M) | Aggregates (4.49M) | **24x speedup** via pre-aggregation |
| **04** | `nlp_features` | Silver (4.97M) | NLP features (896K) | 70.9% variance retained |
| **05** | `build_gold` | 3 silver tables | Gold (896K × 67 cols) | ML-ready dataset |
| **06** | `train_models` | Gold (896K) | 2 UC models | 67.3% R², 97.96% AUC |
| **07** | `score_boroughs` | 55K open complaints | 401 risk scores | 23.8% flagged MEDIUM |

### Notebook Descriptions

#### **01_ingest_bronze.ipynb** - Data Ingestion
* **Purpose:** Load raw CSV from S3 into Bronze Delta table
* **Input:** `s3://civiclens-data/nyc311/nyc311_2020_2023.csv` (5.4M rows)
* **Output:** `civic_lens.bronze.nyc_311_raw`
* **Key Features:** Schema inference, metadata tracking, audit trail

#### **02_clean_silver.ipynb** - Data Cleaning
* **Purpose:** Clean and standardize bronze data
* **Transformations:**
  * Parse timestamps (`created_date`, `closed_date`)
  * Calculate `resolution_days`, `is_open`, `never_resolved` flags
  * Normalize borough names using custom UDF
  * Clean text fields (descriptors + resolutions)
  * Filter invalid coordinates and null critical fields
* **Output:** `civic_lens.silver.nyc_311_cleaned` (4.97M records, 91.2% retention)

#### **03_aggregate_silver.ipynb** - Rolling Feature Engineering ⚡
* **Purpose:** Compute time-based aggregate features for ML
* **Features Generated:**
  * `borough_blackhole_rate`: Rolling 12-month unresolved rate per borough
  * `agency_resolution_rate_hist`: Rolling 12-month agency resolution rate
  * `agency_open_complaints_30d`: Trailing 30-day open complaint volume
* **Performance:** **24x speedup** (2+ hours → <5 minutes) via pre-aggregation strategy
* **Output:** `civic_lens.silver.nyc_borough_agency_agg` (4.49M records)

#### **04_nlp_features.ipynb** - NLP Feature Engineering
* **Purpose:** Extract semantic features from complaint text
* **Techniques:**
  * **Urgency scoring:** 22 keywords (emergency, dangerous, flooding, etc.)
  * **TF-IDF + SVD:** 5,000 terms → 50 dimensions (70.9% variance retained)
  * **LDA topic modeling:** 12 topics extracted
* **Optimization:** Sample-based processing (900K records, 18.1%) to avoid OOM
* **Output:** `civic_lens.silver.nyc_nlp_features` (896K records, 52 features)

#### **05_build_gold.ipynb** - Gold Layer Construction
* **Purpose:** Join silver tables into unified ML dataset
* **Joins:**
  * Base (cleaned) + NLP features + aggregates
  * 896K records × 67 features
* **Output:** `civic_lens.gold.nyc_311_enriched` (ML-ready)

#### **06_train_models.ipynb** - Model Training
* **Purpose:** Train ML models for resolution time prediction
* **Models:**
  * **Gradient Boosting Regressor:** 67.3% R², 97.96% AUC (Champion)
  * **Logistic Regression:** Baseline comparison
* **Target:** `resolution_days` (regression) + binary classification (resolved in 7 days)
* **Output:** Models registered in Unity Catalog via MLflow

#### **07_score_boroughs.ipynb** - Risk Scoring
* **Purpose:** Apply trained models to score open complaints
* **Scoring:**
  * Batch inference on 55,757 open complaints
  * Aggregate to borough-agency level (401 combinations)
  * 3-tier risk classification (High/Medium/Low)
* **Output:** `civic_lens.output.nyc_borough_risk`
* **Key Metric:** 23.8% of borough-agencies flagged as MEDIUM risk

### Key Technical Achievements
* ✅ **5.4M records processed** with 91.2% data retention
* ✅ **24x performance optimization** in aggregate feature engineering
* ✅ **67.3% R² / 97.96% AUC** for resolution prediction
* ✅ **401 risk scores** generated for operational dashboards
* ✅ **70.9% variance** captured in 50-dimensional text embeddings

---

## 🇮🇳 Bangalore Pipeline Summary

**Data Source:** BBMP (Bruhat Bengaluru Mahanagara Palike) Civic Complaints (2020-2025)  
**Scale:** 766K complaints across 6 years  
**Geography:** 199 wards, 32 complaint categories  
**Output:** 4,534 ward-category risk scores

### Pipeline Stages (7 Notebooks)

| Stage | Notebook | Input | Output | Key Achievement |
|-------|----------|-------|--------|-----------------|
| **01** | `ingest_bronze` | JSON (6 files) | Bronze (766K) | Custom JSON parsing |
| **02** | `clean_silver` | Bronze (766K) | Silver (766K) | 100% retention |
| **03** | `aggregate_silver` | Silver (766K) | Aggregates (4.5K) | Ward-category metrics |
| **04** | `nlp_features` | Silver (766K) | NLP features (766K) | TF-IDF + SVD (50D) |
| **05** | `build_gold` | 3 silver tables | Gold (766K × 86 cols) | 100% join coverage |
| **06** | `train_models` | Gold (766K) | 2 UC models | **97.42% accuracy** |
| **07** | `score_wards` | 766K complaints | 4,534 risk scores | Probabilistic scoring |

### Notebook Descriptions

#### **01_ingest_bronze.ipynb** - Data Ingestion
* **Purpose:** Ingest custom JSON format from S3
* **Input:** `s3://civiclens-data/bangalore/bbmp-{year}.json` (2020-2025)
* **Key Challenge:** Parse non-standard JSON: `{fields: [...], records: [...]}`
* **Operations:**
  * Extract field metadata and sanitize column names
  * Union across 6 years with schema drift handling
  * Partition by `source_year`
* **Output:** `civic_lens.bronze.bbmp_complaints_raw` (766,648 records)

#### **02_clean_silver.ipynb** - Data Cleaning & Enrichment
* **Purpose:** Transform bronze into analysis-ready silver layer
* **Transformations:**
  * Parse `Grievance_Date` timestamps, extract temporal features
  * **Outcome label engineering:** 3-class (Resolved, Closed, Rejected)
  * Extract `staff_dept` from "Name/Dept" format
  * Compute remark quality flags (`remark_length`, `remark_is_boilerplate`)
  * Normalize ward names using `geo_utils.normalize_ward_name()`
* **Output:** `civic_lens.silver.bbmp_complaints_clean` (766K records, 19 columns)
* **Quality:** 100% retention, 88.4% boilerplate rate detected

#### **03_aggregate_silver.ipynb** - Ward-Category Aggregations
* **Purpose:** Compute service quality metrics by ward × category
* **Dimensions:** 199 wards × 32 categories = 4,534 combinations
* **Metrics:**
  * `rejection_rate`, `boilerplate_rate`
  * `total_complaints`, `total_complaints_30d`, `open_complaints_30d`
  * `avg_remark_length`, `unique_depts_handling`
* **Output:** `civic_lens.silver.bbmp_ward_category_agg` (4,534 records)
* **Key Finding:** 16.3% average rejection rate, 65.7% boilerplate rate

#### **04_nlp_features.ipynb** - NLP Feature Engineering
* **Purpose:** Extract semantic features from staff remarks
* **Techniques:**
  * **Urgency score:** 35+ keywords (Indian context: dengue, sewage, overflow)
  * **TF-IDF vectorization:** 5,000 terms, smart sampling (non-boilerplate focus)
  * **Truncated SVD:** 5,000 → 50 dimensions
* **Optimization:** `mapInPandas` for distributed batch transformation (766K docs)
* **Output:** `civic_lens.silver.bbmp_nlp_features` (766K records, 57 columns)

#### **05_build_gold.ipynb** - Gold Layer Construction
* **Purpose:** Create unified ML-ready table
* **Joins:**
  * Base (19 cols) + NLP (51 cols) + Ward aggregates (8 cols) + Derived flags (8 cols)
  * ~86 total features
* **Join Coverage:** 100% for NLP (perfect 1:1), ~99% for aggregates
* **Output:** `civic_lens.gold.bbmp_complaints_enriched` (766K × 86 features)

#### **06_train_models.ipynb** - ML Model Training
* **Purpose:** Train multi-class outcome classification
* **Target:** `outcome_label` (0=Resolved, 1=Closed, 2=Rejected)
* **Class Distribution:**
  * Class 0: 6,953 (0.9%)
  * Class 1: 702,410 (94.6%) - **severe imbalance**
  * Class 2: 32,853 (4.4%)
* **Models:**
  * **XGBoost Multi-Class:** 97.42% test accuracy, 0.8273 Macro F1 (Champion ⭐)
  * **Logistic Regression:** 95.17% test accuracy, 0.7613 Macro F1
* **Class Imbalance Handling:** Stratified split + sample weighting
* **Output:** Models logged to MLflow with full tracking

#### **07_score_wards.ipynb** - Ward Risk Scoring
* **Purpose:** Apply trained model for ward-level risk assessment
* **Scoring Pipeline:**
  * Batch inference on 766K complaints using `mapInPandas`
  * **Key Innovation:** Uses **average rejection probability** (soft predictions) instead of hard class counts
  * Solves "zero-wall" problem from severe class imbalance
  * Aggregate to 4,534 ward-category combinations
* **Risk Tiers:**
  * High Risk: 933 (20.6%) - rejection_risk_score > 10%
  * Medium Risk: 222 (4.9%) - 5-10%
  * Low Risk: 3,379 (74.5%) - <5%
* **Output:** `civic_lens.output.bangalore_ward_risk` (4,534 scores)

### Key Technical Achievements
* ✅ **766K complaints** analyzed across 6 years (2020-2025)
* ✅ **97.42% classification accuracy** with XGBoost
* ✅ **100% data retention** through entire pipeline
* ✅ **Probabilistic risk scoring** handles sparse cells elegantly
* ✅ **4,534 ward-category scores** for granular civic insights
* ✅ **50-dimensional semantic embeddings** via TF-IDF + SVD

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|------------|
| **Cloud Platform** | AWS |
| **Data Lake** | Delta Lake (S3-backed) |
| **Compute** | Databricks Serverless Spark |
| **Languages** | PySpark (Python 3.12) |
| **ML Frameworks** | XGBoost, scikit-learn, Gradient Boosting |
| **ML Ops** | MLflow (tracking, registry, deployment) |
| **NLP** | TF-IDF, Truncated SVD, LDA |
| **Data Catalog** | Unity Catalog |
| **Orchestration** | Databricks Workflows |

---

## 📊 Side-by-Side Comparison

| Metric | NYC Pipeline | Bangalore Pipeline |
|--------|--------------|-------------------|
| **Data Source** | NYC 311 Requests | BBMP Civic Complaints |
| **Time Period** | 2022-2023 | 2020-2025 |
| **Raw Records** | 5.4M | 766K |
| **Clean Records** | 4.97M (91.2%) | 766K (100%) |
| **Geography** | 5 boroughs | 199 wards |
| **Categories** | Multiple agencies | 32 complaint types |
| **ML Target** | Resolution days (regression) | Outcome label (3-class) |
| **Champion Model** | Gradient Boosting (67.3% R²) | XGBoost (97.42% accuracy) |
| **Risk Scores** | 401 borough-agency scores | 4,534 ward-category scores |
| **Special Optimization** | 24x speedup (aggregations) | Probabilistic risk scoring |
| **Notebook Count** | 7 stages | 7 stages |

---

## 📈 Business Impact

### NYC
* **55,757 open complaints** scored for prioritization
* **401 borough-agency combinations** monitored
* **23.8% MEDIUM risk** areas flagged for intervention
* **97.96% AUC** for binary classification (resolved in 7 days)

### Bangalore
* **766,648 complaints** analyzed historically
* **4,534 ward-category combinations** scored
* **20.6% HIGH risk** areas identified (933 combinations)
* **65.7% boilerplate rate** reveals service quality opportunity

---

## 🎯 Common Pipeline Patterns

Both pipelines follow these production best practices:

### 1. **Medallion Architecture**
* **Bronze:** Raw data preservation with minimal transformation
* **Silver:** Cleaned, validated, enriched data ready for analytics
* **Gold:** ML-ready unified tables with zero nulls in critical features

### 2. **Feature Engineering**
* **Temporal features:** Extract date components, rolling windows
* **Text features:** TF-IDF, SVD, topic modeling, urgency scoring
* **Aggregate features:** Rolling metrics, rate calculations
* **Quality flags:** Boolean indicators for data quality issues

### 3. **ML Pipeline**
* **Class imbalance handling:** Stratified splits, sample weighting
* **Model comparison:** Multiple algorithms with MLflow tracking
* **Probabilistic scoring:** Use soft predictions to handle sparse data
* **Distributed inference:** Batch scoring via `mapInPandas`

### 4. **Data Quality**
* **Schema validation:** Enforce data types, null constraints
* **Join coverage analysis:** Track % of successful joins
* **Null checks:** Zero nulls in critical ML features
* **Duplicate detection:** Unique ID validation

### 5. **Performance Optimization**
* **Partitioning:** By date/year for query pruning
* **Pre-aggregation:** Reduce data volume before expensive operations
* **Sampling:** Use representative samples for NLP fitting
* **Distributed processing:** `mapInPandas` for scalable transformations

---

## 🚀 Getting Started

### Navigate to City-Specific Pipelines

* **NYC Pipeline:** See [nyc/README.md](nyc/README.md) for detailed documentation
* **Bangalore Pipeline:** See [bangalore/README.md](bangalore/README.md) for detailed documentation

### Run Notebooks in Sequence

Each pipeline has 7 notebooks designed to run sequentially:

```bash
# Recommended execution order (both cities follow same pattern)
01_ingest_bronze.ipynb      # Start here - load raw data
02_clean_silver.ipynb       # Clean and standardize
03_aggregate_silver.ipynb   # Build aggregate features
04_nlp_features.ipynb       # Extract text features
05_build_gold.ipynb         # Join to ML table
06_train_models.ipynb       # Train ML models
07_score_<geo>.ipynb        # Generate risk scores
```

### Unity Catalog Table Structure

```
civic_lens/
├── bronze/
│   ├── nyc_311_raw
│   └── bbmp_complaints_raw
├── silver/
│   ├── nyc_311_cleaned
│   ├── nyc_borough_agency_agg
│   ├── nyc_nlp_features
│   ├── bbmp_complaints_clean
│   ├── bbmp_ward_category_agg
│   └── bbmp_nlp_features
├── gold/
│   ├── nyc_311_enriched
│   └── bbmp_complaints_enriched
└── output/
    ├── nyc_borough_risk
    └── bangalore_ward_risk
```

---

## 📝 Future Enhancements

### Short-Term
* [ ] Real-time inference endpoints for live complaint scoring
* [ ] Unified dashboards with ward/borough heat maps
* [ ] Automated alerts for high-risk areas (Slack, email)
* [ ] A/B testing framework for intervention strategies

### Medium-Term
* [ ] Time-series forecasting for complaint volumes
* [ ] Geospatial clustering analysis using ward/borough polygons
* [ ] SHAP values for model interpretability
* [ ] Multi-city expansion: Delhi, Mumbai, Chennai, Los Angeles

### Long-Term
* [ ] Transformer-based embeddings (BERT) for text analysis
* [ ] Causal inference for intervention impact measurement
* [ ] Recommendation systems for resolution strategies
* [ ] Public API for civic transparency apps

---

## 🤝 Contributing

This is a portfolio project demonstrating production-grade ML engineering. For questions or collaboration:

**Author:** Pawan Virat  
**Email:** pawanvirat32@gmail.com  
**LinkedIn:** [linkedin.com/in/pawanvirat](https://linkedin.com/in/pawanvirat)

---

## 📜 License & Attribution

* **NYC Data:** NYC Open Data Portal - 311 Service Requests
* **Bangalore Data:** BBMP (Bruhat Bengaluru Mahanagara Palike) public records
* **License:** Educational & portfolio use only
* **Privacy:** No PII (personally identifiable information) used

---

## 📌 Citation

If referencing this project:

```
Virat, P. (2025). CivicLens: Multi-City Civic Complaints Analytics Platform.
End-to-End ML Pipeline for NYC (5.4M) and Bangalore (766K) Complaints.
Databricks Lakehouse Platform. 97%+ prediction accuracy, production-ready risk scoring.
```

---

**Last Updated:** June 2026  
**Platform Version:** 1.0  
**Status:** ✅ Production-Ready

---
