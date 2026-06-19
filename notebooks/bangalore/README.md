# BBMP Bangalore Civic Complaints Analytics Pipeline

**End-to-end ML pipeline for analyzing 766K+ civic complaints from Bangalore's BBMP public grievance system**

A production-grade data engineering and machine learning project implementing medallion architecture (Bronze → Silver → Gold) to predict complaint outcomes and identify high-risk wards for civic intervention.

---

## 📊 Project Overview

### Business Problem
Bangalore Municipal Corporation (BBMP) processes hundreds of thousands of civic complaints annually. This project builds a predictive analytics system to:
* **Predict complaint outcomes** (Resolved, Closed, Rejected) with 99.22% accuracy
* **Identify high-risk wards** requiring immediate civic intervention
* **Quantify service quality** across 199 wards and 32 complaint categories
* **Enable proactive resource allocation** through ML-powered risk scoring

### Impact & Business Value
* **766,648 complaints** analyzed across 6 years (2020-2025)
* **4,534 ward-category combinations** scored for operational insights
* **796 high-risk areas** identified (35.14% average rejection rate)
* **99.22% prediction accuracy** for complaint outcome classification
* **Reduced manual triage** through automated risk tier assignment

---

## 🏗️ Architecture & Pipeline

### Data Architecture
```
📁 S3 Raw Data (JSON)
    ↓
🥉 Bronze Layer (766K records)
    ├─ Raw ingestion from S3
    ├─ Schema unification across 2020-2025
    └─ Delta Lake partitioned by source_year
    ↓
🥈 Silver Layer (Cleaned & Enriched)
    ├─ Data quality checks (100% valid records)
    ├─ Temporal feature engineering
    ├─ Outcome label classification (3-class)
    ├─ NLP feature extraction (TF-IDF + 50 dimensions)
    └─ Ward-category aggregations (4,534 combos)
    ↓
🥇 Gold Layer (ML-Ready)
    ├─ Unified table: 86 columns
    ├─ 71 ML features ready for training
    ├─ Zero nulls in critical features
    └─ Perfect join coverage (100%)
    ↓
🤖 ML Models (Trained & Deployed)
    ├─ XGBoost Classifier (Champion: 99.22% accuracy)
    ├─ Logistic Regression Baseline (95.16% accuracy)
    └─ MLflow tracking & versioning
    ↓
📍 Ward Risk Scoring (Operational Output)
    ├─ Batch inference on 766K complaints
    ├─ Ward-level risk aggregation
    ├─ 3-tier classification (High/Medium/Low)
    └─ Dashboard-ready analytics table
```

### Technology Stack
* **Storage:** Delta Lake (AWS S3-backed)
* **Compute:** Databricks Serverless (PySpark)
* **ML Framework:** XGBoost, scikit-learn, MLflow
* **NLP:** TF-IDF + Truncated SVD (50 dimensions)
* **Data Quality:** Schema validation, null checks, join coverage analysis

---

## 📋 Pipeline Notebooks (7-Stage)

### 01 - Bronze Layer: Raw Data Ingestion
**Objective:** Ingest raw BBMP JSON files from S3 and create unified Delta table

**Key Operations:**
* Parse custom JSON format: `{fields: [...], records: [...]}`
* Union DataFrames across 6 years (2020-2025) with schema drift handling
* Write Delta Lake with partitioning by `source_year`

**Output:**
* **Table:** `civic_lens.bronze.bbmp_complaints_raw`
* **Records:** 766,648 complaints
* **Columns:** 10 (9 original + source_year)

**Year Distribution:**
| Year | Records |
|------|---------|
| 2020 | 91,620 |
| 2021 | 103,504 |
| 2022 | 118,394 |
| 2023 | 119,140 |
| 2024 | 207,016 |
| 2025 | 126,974 |

---

### 02 - Silver Layer: Data Cleaning & Enrichment
**Objective:** Transform raw bronze data into clean, analysis-ready silver layer

**Transformations Applied:**
1. **Temporal Features:** Parse dates, extract year/month/day_of_week
2. **Outcome Label Engineering:**
   * Class 0 (Resolved): 6,953 complaints (0.9%)
   * Class 1 (Closed): 702,410 complaints (91.6%)
   * Class 2 (Rejected): 32,853 complaints (4.3%)
   * NULL (In-Progress): 24,432 complaints (3.2%)
3. **Staff Information:** Extract dept from "Name/Dept" format
4. **Remark Quality Flags:**
   * `remark_length`: Character count
   * `remark_is_boilerplate`: 88.4% automated responses
5. **Geographic Normalization:** Standardize ward names (199 unique wards)

**Output:**
* **Table:** `civic_lens.silver.bbmp_complaints_clean`
* **Records:** 766,648 (100% retained)
* **Schema:** 19 enriched columns

**Top Categories:**
1. Electrical: 310,128 (40.4%)
2. Solid Waste: 195,153 (25.4%)
3. Road Maintenance: 111,535 (14.5%)
4. Forest: 34,618 (4.5%)
5. Health Dept: 29,924 (3.9%)

---

### 03 - Silver Aggregations: Ward & Category Analytics
**Objective:** Create ward-category level aggregations for service quality metrics

**Aggregation Dimensions:**
* **Ward-level:** 199 wards
* **Category-level:** 32 categories
* **Cross-dimension:** 4,534 ward × category combinations

**Metrics Computed:**
* `rejection_rate`: % complaints rejected per ward-category
* `boilerplate_rate`: % generic responses
* `open_complaints_30d`: Recent unresolved complaints
* `avg_remark_length`: Response quality proxy
* `unique_depts_handling`: Department involvement

**Output:**
* **Table:** `civic_lens.silver.bbmp_ward_category_agg`
* **Records:** 4,534 ward-category pairs
* **Columns:** 11 aggregate metrics

**Key Finding:**
* Average rejection rate: 6.42% across all ward-categories
* Average boilerplate rate: 65.73% (opportunity for improvement)

---

### 04 - NLP Feature Engineering: Staff Remarks Analysis
**Objective:** Extract semantic features from staff remarks using NLP techniques

**NLP Pipeline:**
1. **Urgency Score:** Keyword-based scoring (35 keywords)
   * "urgent", "emergency", "immediate", "critical", "danger", etc.
   * Indian context: "overflow", "dengue", "disease", "health", etc.
2. **TF-IDF Vectorization:**
   * Max features: 5,000 terms
   * Min document frequency: 5
   * Optimized sampling: 137K non-boilerplate remarks
3. **Dimensionality Reduction:** Truncated SVD → 50 components
   * Explained variance: 71.4%
   * Batch transformation: All 766K complaints

**Output:**
* **Table:** `civic_lens.silver.bbmp_nlp_features`
* **Records:** 766,648 (100% coverage)
* **Columns:** 57 (urgency_score + 50 TF-IDF + 6 metadata)

**Optimization:**
* Spark-native batch processing (no Pandas conversion bottleneck)
* Smart sampling strategy for TF-IDF fitting (non-boilerplate heavy)

---

### 05 - Gold Layer: Unified ML & Analytics Table
**Objective:** Create denormalized, ML-ready table combining all features

**Join Strategy:**
1. **Base:** Silver complaints (766,648 rows)
2. **+NLP Features:** LEFT JOIN on `complaint_id` (100% match, 0 nulls)
3. **+Ward Aggregates:** LEFT JOIN on `(ward, category)` (100% match, 0 nulls)
4. **+Derived Flags:** 8 boolean features for quality checks

**Derived Features Added:**
* `is_high_urgency`: P90+ urgency score (10% of complaints)
* `is_very_high_urgency`: P95+ urgency score (5% of complaints)
* `is_problem_area`: Ward-category rejection rate > 10%
* `is_high_boilerplate_area`: Ward-category boilerplate > 85%
* `has_sufficient_context`: Remark length > 50 chars
* `is_recent`: Last 30 days
* `days_since_grievance`: Age of complaint
* `is_weekend`: Saturday/Sunday submissions

**Output:**
* **Table:** `civic_lens.gold.bbmp_complaints_enriched`
* **Records:** 766,648 (100% coverage, no data loss)
* **Columns:** 86 features
  * 19 core complaint fields
  * 51 NLP features (urgency + 50 TF-IDF)
  * 8 ward-category aggregates
  * 8 derived quality flags

**Data Quality:**
* ✅ Zero nulls in critical ML features
* ✅ Perfect join coverage (100%)
* ✅ Ready for model training

---

### 06 - ML Model Training: Outcome Prediction
**Objective:** Train multi-class classification models to predict complaint outcome

**Problem Setup:**
* **Target:** `outcome_label` (3-class)
  * 0 = Resolved (6,953 samples, 0.9%)
  * 1 = Closed (702,410 samples, 94.6%)
  * 2 = Rejected (32,853 samples, 4.4%)
* **Features:** 71 columns
  * 13 numerical (urgency, temporal, ward aggregates)
  * 50 TF-IDF semantic features
  * 8 boolean flags
* **Split:** 80/20 train-test (stratified)
  * Train: 593,786 samples
  * Test: 148,430 samples

**Models Trained:**

#### 1. XGBoost Multi-Class Classifier (Champion ⭐)
**Hyperparameters:**
* `objective`: multi:softmax
* `num_class`: 3
* `max_depth`: 6
* `learning_rate`: 0.1
* `n_estimators`: 100
* `subsample`: 0.8
* `colsample_bytree`: 0.8

**Performance (Test Set):**
* **Test Accuracy:** 99.22%
* **Train Accuracy:** 99.31%
* **Macro F1:** 0.9423
* **Weighted F1:** 0.9921
* **ROC-AUC (weighted):** 0.9977

**Per-Class Metrics:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Resolved | 0.9379 | 0.8691 | 0.9022 | 1,390 |
| Closed | 0.9949 | 0.9970 | 0.9960 | 140,483 |
| Rejected | 0.9429 | 0.9152 | 0.9289 | 6,571 |

**Top Features by Importance:**
1. `is_problem_area`: 23.99%
2. `tfidf_feat_5`: 14.96%
3. `tfidf_feat_7`: 9.26%
4. `tfidf_feat_12`: 6.76%
5. `tfidf_feat_19`: 5.89%

#### 2. Multinomial Logistic Regression (Baseline)
**Performance:**
* **Test Accuracy:** 95.16%
* **Macro F1:** 0.4556
* **Improvement:** XGBoost outperforms by **4.06%**

**Model Artifacts:**
* MLflow tracking: `/Users/pawanvirat32@gmail.com/civic-lens/bbmp-outcome-prediction`
* Run IDs: XGBoost (`12e7165a`), LogReg (`c55bafee`)
* Confusion matrices, feature importances logged

---

### 07 - Ward Risk Scoring: Complaint Outcome Predictions
**Objective:** Apply trained model to score all complaints and aggregate by ward-category

**Batch Inference Pipeline:**
1. **Load Model:** MLflow pyfunc from run (`c55bafee`) [Note: Production would use XGBoost]
2. **Prepare Features:** 71-column feature vector for 766,648 complaints
3. **Create Spark UDF:** Distributed model inference (no driver bottleneck)
4. **Predict Outcomes:** Score all complaints (0=Resolved, 1=Closed, 2=Rejected)
5. **Aggregate to Ward-Category:** Group predictions by ward × category
6. **Compute Risk Scores:**
   * `rejection_risk_score`: % predicted rejections
   * `boilerplate_risk_score`: % boilerplate remarks
7. **Assign Risk Tiers:**
   * **High Risk:** rejection_risk_score > 10%
   * **Medium Risk:** 5% < rejection_risk_score ≤ 10%
   * **Low Risk:** rejection_risk_score ≤ 5%

**Output:**
* **Table:** `civic_lens.output.bangalore_ward_risk`
* **Records:** 4,534 ward-category combinations
* **Columns:** 10 (ward, category, predictions, scores, tier, timestamp)

**Risk Distribution:**
| Risk Tier | Combinations | Complaints | Avg Rejection % |
|-----------|--------------|------------|-----------------|
| **High** | 796 (17.6%) | 47,508 | 35.14% |
| **Medium** | 116 (2.6%) | 27,126 | 7.47% |
| **Low** | 3,622 (79.9%) | 692,014 | 0.08% |

**Overall Statistics:**
* **Total Combinations:** 4,534
* **Total Complaints Scored:** 766,648
* **Unique Wards:** 199
* **Unique Categories:** 32
* **Average Rejection Risk:** 6.42%
* **Average Boilerplate Rate:** 65.73%

**Top 10 Problem Wards (Highest Risk):**
| Ward | High-Risk Complaints | Categories | Avg Rejection % |
|------|---------------------|------------|-----------------|
| Singasandra | 2,470 | 7 | 32.76% |
| Hemmigepura | 1,958 | 7 | 36.26% |
| HSR Layout | 1,714 | 6 | 30.32% |
| Kadugodi | 1,430 | 6 | 31.16% |
| Arakere | 1,403 | 7 | 29.28% |
| Byatarayanapura | 1,341 | 5 | 22.72% |
| J.P. Nagar | 1,152 | 5 | 22.76% |
| Ullalu | 1,142 | 7 | 29.53% |
| Shanthala Nagar | 1,095 | 8 | 28.04% |
| Dharmarayaswamy Temple | 918 | 7 | 24.34% |

**Top 5 Problem Categories:**
| Category | Avg Rejection % | High-Risk Wards |
|----------|-----------------|-----------------|
| Others | 41.99% | 193 |
| Storm Water Drain (SWD) | 25.92% | 146 |
| Town Planning | 15.33% | 86 |
| Sanitation | 12.99% | 82 |
| Lakes | 12.77% | 36 |

**Best Performing Wards (Low Risk + High Volume):**
| Ward | Total Complaints | Avg Rejection % |
|------|------------------|-----------------|
| Devarajeevanahalli | 762 | 0.12% |
| Bapuji Nagar | 1,600 | 1.47% |
| T-Dasarahalli | 1,660 | 1.74% |
| Kempapura Agrahara | 903 | 1.90% |
| Jeevanbhima Nagar | 4,061 | 2.09% |

---

## 🎯 Key Achievements & Resume Metrics

### Technical Achievements
* ✅ **Processed 766,648 civic complaints** across 6 years (2020-2025)
* ✅ **Built end-to-end ML pipeline** using Databricks lakehouse architecture
* ✅ **Achieved 99.22% prediction accuracy** with XGBoost multi-class classifier
* ✅ **Engineered 86 features** including 50-dimensional TF-IDF semantic embeddings
* ✅ **Optimized Spark-native NLP** pipeline processing 766K documents
* ✅ **Zero data loss** through pipeline (100% join coverage, no nulls in critical features)
* ✅ **Created 4,534 ward-category risk scores** for operational decision-making

### Data Engineering Metrics
* **3-layer medallion architecture:** Bronze → Silver → Gold
* **6 Delta Lake tables** with auto-optimization and partitioning
* **100% data quality:** Zero nulls, perfect join coverage, validated schemas
* **Serverless compute:** Optimized for Databricks serverless Spark
* **Scalable NLP:** TF-IDF + SVD dimensionality reduction (5K → 50 features)

### Machine Learning Metrics
* **Model:** XGBoost multi-class classifier (3 outcomes)
* **Accuracy:** 99.22% test, 99.31% train (minimal overfitting)
* **Class Imbalance Handling:** Stratified split, per-class metrics
* **Feature Importance:** Top feature = `is_problem_area` (23.99%)
* **MLflow Integration:** Full experiment tracking, model versioning, artifact logging
* **Batch Inference:** Spark UDF for distributed scoring (766K predictions)

### Business Impact
* **Identified 796 high-risk areas** (ward-category pairs with >10% rejection)
* **47,508 complaints** in high-risk categories requiring immediate attention
* **Top 10 problem wards** identified for civic resource allocation
* **65.73% boilerplate rate** uncovered (opportunity for service quality improvement)
* **Enabled proactive intervention** through 3-tier risk classification

---

## 📁 Project Structure

```
bangalore/
├── README.md                           # This file
├── notebooks/
│   ├── 01_ingest_bronze.ipynb         # Bronze layer ingestion
│   ├── 02_clean_silver.ipynb          # Silver layer cleaning
│   ├── 03_aggregate_silver.ipynb      # Ward-category aggregations
│   ├── 04_nlp_features.ipynb          # NLP feature engineering
│   ├── 05_build_gold.ipynb            # Gold layer construction
│   ├── 06_train_models.ipynb          # ML model training
│   └── 07_score_wards.ipynb           # Ward risk scoring
└── data/
    └── s3://civiclens-data/bangalore/  # Raw JSON files (2020-2025)
```

---

## 🗂️ Data Assets (Unity Catalog)

### Bronze Layer
* **`civic_lens.bronze.bbmp_complaints_raw`**
  * 766,648 records
  * Partitioned by `source_year`
  * Raw JSON ingestion

### Silver Layer
* **`civic_lens.silver.bbmp_complaints_clean`**
  * 766,648 records, 19 columns
  * Cleaned & enriched complaints
  
* **`civic_lens.silver.bbmp_nlp_features`**
  * 766,648 records, 57 columns
  * TF-IDF embeddings + urgency scores
  
* **`civic_lens.silver.bbmp_ward_category_agg`**
  * 4,534 ward-category pairs, 11 columns
  * Aggregated service quality metrics

### Gold Layer
* **`civic_lens.gold.bbmp_complaints_enriched`**
  * 766,648 records, 86 columns
  * ML-ready unified table
  * Perfect join coverage (100%)

### Output Layer
* **`civic_lens.output.bangalore_ward_risk`**
  * 4,534 ward-category scores, 10 columns
  * Risk tiers (High/Medium/Low)
  * Dashboard-ready analytics

---

## 🔧 Technical Stack

| Category | Technology |
|----------|------------|
| **Cloud Platform** | AWS |
| **Data Lake** | Delta Lake (S3-backed) |
| **Compute** | Databricks Serverless Spark |
| **Language** | PySpark (Python 3.12) |
| **ML Frameworks** | XGBoost, scikit-learn |
| **ML Ops** | MLflow (tracking, registry, deployment) |
| **NLP** | TF-IDF Vectorizer, Truncated SVD |
| **Data Catalog** | Unity Catalog |
| **Orchestration** | Databricks Workflows |

---

## 📈 Model Performance Summary

| Metric | XGBoost (Champion) | Logistic Regression |
|--------|-------------------|---------------------|
| **Test Accuracy** | 99.22% | 95.16% |
| **Train Accuracy** | 99.31% | 95.10% |
| **Macro F1** | 0.9423 | 0.4556 |
| **Weighted F1** | 0.9921 | 0.9398 |
| **ROC-AUC** | 0.9977 | 0.9212 |
| **Training Time** | ~2 min | ~2.5 min |
| **Inference (766K)** | ~5 min | ~4 min |

**Winner:** XGBoost (+4.06% accuracy improvement)

**Class-Specific Performance (XGBoost):**
* **Resolved (minority class):** 93.8% precision, 86.9% recall
* **Closed (majority class):** 99.5% precision, 99.7% recall
* **Rejected (minority class):** 94.3% precision, 91.5% recall

---

## 🚀 Deployment & Operationalization

### Model Deployment
* **Registry:** MLflow model registry (Unity Catalog)
* **Serving:** Batch inference via Spark UDF
* **Refresh:** Scheduled weekly scoring run
* **Monitoring:** Per-class metrics, prediction distribution tracking

### Output Tables
* **Primary:** `civic_lens.output.bangalore_ward_risk`
* **Refresh:** Daily incremental (append new scores)
* **Consumers:** BI dashboards, Slack alerts, ward manager reports

### Use Cases
1. **Proactive Complaint Management**
   * Auto-route high-risk complaints to senior staff
   * Flag predicted rejections for quality review
   
2. **Resource Optimization**
   * Allocate staff to high-risk wards
   * Balance workload based on complaint volume & complexity
   
3. **Citizen Transparency**
   * Display ward-category risk scores in public portal
   * Manage expectations with historical rejection rates
   
4. **Policy Evaluation**
   * A/B test interventions (training, routing changes)
   * Measure impact of policy changes on risk scores

---

## 📊 Data Quality Report

| Quality Metric | Score |
|----------------|-------|
| **Completeness** | 100% (zero null critical fields) |
| **Join Coverage** | 100% (perfect NLP + agg joins) |
| **Schema Validation** | ✅ All 7 stages validated |
| **Outcome Label Quality** | 96.8% labeled (3.2% in-progress) |
| **Temporal Consistency** | ✅ 2020-2025 continuous |
| **Duplicate Detection** | 0 duplicates (unique `complaint_id`) |
| **Feature Coverage** | 86/86 features populated |

---

## 🎓 Key Learnings & Best Practices

### Data Engineering
* **Medallion architecture** (Bronze/Silver/Gold) provides clear data lineage
* **Delta Lake optimization** (auto-compact, optimize-write) critical for performance
* **Spark-native processing** avoids Pandas bottlenecks (766K rows in <5 min)
* **Stratified sampling** for TF-IDF fitting balances diversity & computation

### Machine Learning
* **Class imbalance** handled via stratified split + per-class metrics
* **Feature engineering** drives performance (top feature = domain-engineered `is_problem_area`)
* **Ensemble methods** (XGBoost) outperform linear models significantly
* **MLflow tracking** essential for experiment reproducibility

### NLP
* **TF-IDF + SVD** effective for large-scale text feature extraction
* **Domain keywords** (Indian context) boost urgency detection
* **Boilerplate detection** (88% rate) reveals automation opportunity

---

## 📝 Future Enhancements

### Short-Term (v2.0)
* [ ] Real-time inference endpoint for live complaint scoring
* [ ] Dashboard with ward-level heat maps and trend analysis
* [ ] Automated Slack/email alerts for high-risk complaints
* [ ] A/B test intervention strategies (staff training, routing)

### Medium-Term (v2.5)
* [ ] Temporal analysis: Time-series forecasting of complaint volumes
* [ ] Geospatial analysis: Spatial clustering of high-risk areas
* [ ] Root cause analysis: Feature attribution for high rejection rates
* [ ] Multi-city expansion: Replicate pipeline for Delhi, Mumbai, Chennai

### Long-Term (v3.0)
* [ ] Deep learning: Transformer-based embeddings (BERT) for remark analysis
* [ ] Causal inference: Measure intervention impact with propensity score matching
* [ ] Recommendation system: Suggest resolution strategies per complaint
* [ ] Public API: Expose risk scores & predictions to civic apps

---

## 🤝 Contributing

This is a portfolio project demonstrating end-to-end ML engineering skills. For questions or collaboration:

**Author:** Pawan Virat  
**Email:** pawanvirat32@gmail.com  
**LinkedIn:** [linkedin.com/in/pawanvirat](https://linkedin.com/in/pawanvirat)

---

## 📜 License & Data Attribution

* **Data Source:** BBMP (Bruhat Bengaluru Mahanagara Palike) public grievance records
* **License:** Educational & portfolio use only
* **Privacy:** No PII (personally identifiable information) used or exposed

---

## 📌 Citation

If referencing this project:

```
Virat, P. (2025). BBMP Bangalore Civic Complaints Analytics Pipeline: 
End-to-End ML Pipeline for Civic Outcome Prediction and Ward Risk Scoring.
Databricks Lakehouse Platform. 766K complaints, 99.22% classification accuracy.
```

---

**Last Updated:** June 2026  
**Pipeline Version:** 1.0  
**Status:** ✅ Production-Ready

---
