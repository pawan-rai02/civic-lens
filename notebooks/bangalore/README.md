# BBMP Bangalore Civic Complaints Analytics Pipeline

**End-to-end ML pipeline for analyzing 766K+ civic complaints from Bangalore's BBMP public grievance system**

A production-grade data engineering and machine learning project implementing medallion architecture (Bronze → Silver → Gold) to predict complaint outcomes and identify high-risk wards for civic intervention.

---

## 📊 Project Overview

### Business Problem
Bangalore Municipal Corporation (BBMP) processes hundreds of thousands of civic complaints annually. This project builds a predictive analytics system to:
* **Predict complaint outcomes** (Resolved, Closed, Rejected) with 97.42% accuracy
* **Identify high-risk wards** requiring immediate civic intervention
* **Quantify service quality** across 199 wards and 32 complaint categories
* **Enable proactive resource allocation** through ML-powered risk scoring

### Impact & Business Value
* **766,648 complaints** analyzed across 6 years (2020-2025)
* **4,534 ward-category combinations** scored for operational insights
* **97.42% prediction accuracy** for complaint outcome classification (XGBoost)
* **Probabilistic risk scoring** using soft predictions to avoid sparse-cell bias
* **Reduced manual triage** through automated risk tier assignment

---

## 🏗️ Architecture & Pipeline

### Data Architecture
```
📁 S3 Raw Data (JSON)
    ↓
🥉 Bronze Layer (766K records)
    ├─ Raw ingestion from S3
    ├─ Custom JSON parsing {fields: [...], records: [...]}
    ├─ Schema unification across 2020-2025
    └─ Delta Lake partitioned by source_year
    ↓
🥈 Silver Layer (Cleaned & Enriched)
    ├─ Data quality checks (100% valid records)
    ├─ Temporal feature engineering
    ├─ Outcome label classification (3-class)
    ├─ NLP feature extraction (TF-IDF + SVD → 50 dimensions)
    ├─ Ward name normalization using geo_utils
    └─ Ward-category aggregations (4,534 combos)
    ↓
🥇 Gold Layer (ML-Ready)
    ├─ Unified table: ~86 columns
    ├─ 71 ML features ready for training
    ├─ Zero nulls in critical features
    └─ Perfect join coverage (100%)
    ↓
🤖 ML Models (Trained & Deployed)
    ├─ XGBoost Classifier (Champion: 97.42% accuracy)
    ├─ Logistic Regression Baseline (95.17% accuracy)
    └─ MLflow tracking & versioning
    ↓
📍 Ward Risk Scoring (Operational Output)
    ├─ Batch inference on 766K complaints
    ├─ Probabilistic risk scoring (avg rejection probability)
    ├─ Ward-level risk aggregation
    ├─ 3-tier classification (High/Medium/Low)
    └─ Dashboard-ready analytics table
```

### Technology Stack
* **Storage:** Delta Lake (AWS S3-backed)
* **Compute:** Databricks Serverless (PySpark)
* **ML Framework:** XGBoost, scikit-learn, MLflow
* **NLP:** TF-IDF Vectorizer + Truncated SVD (50 dimensions)
* **Geospatial:** Custom geo_utils for ward name normalization
* **Data Quality:** Schema validation, null checks, join coverage analysis

---

## 📋 Pipeline Notebooks (7-Stage)

### 01 - Bronze Layer: Raw Data Ingestion
**Objective:** Ingest raw BBMP JSON files from S3 and create unified Delta table

**Key Operations:**
* Parse custom JSON format: `{fields: [...], records: [...]}`
* Sanitize column names (remove special characters, spaces → underscores)
* Union DataFrames across 6 years (2020-2025) with `unionByName(allowMissingColumns=True)`
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
1. **Temporal Features:** Parse dates using `to_timestamp`, extract year/month/day_of_week
2. **Outcome Label Engineering:**
   * Class 0 (Resolved): 6,953 complaints (0.9%)
   * Class 1 (Closed): 702,410 complaints (91.6%)
   * Class 2 (Rejected): 32,853 complaints (4.3%)
   * NULL (In-Progress): 24,432 complaints (3.2%)
3. **Staff Information:** Extract `staff_dept` from "Name/Dept" format using `split()`
4. **Remark Quality Flags:**
   * `remark_length`: Character count
   * `remark_is_boilerplate`: Boolean flag (≤20 chars or common phrases like "attended", "closed")
   * Boilerplate rate: 88.4%
5. **Geographic Normalization:** Standardize ward names using `geo_utils.normalize_ward_name()` (199 unique wards)

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

**Top Staff Departments:**
1. AEE: 471,629
2. JHI: 54,750
3. AE: 51,422
4. Customer Support: 37,850

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
* `total_complaints`: All-time complaint count
* `total_complaints_30d`: Recent complaints (trailing 30 days)
* `open_complaints_30d`: Recent unresolved complaints
* `avg_remark_length`: Response quality proxy
* `unique_depts_handling`: Department involvement

**Output:**
* **Table:** `civic_lens.silver.bbmp_ward_category_agg`
* **Records:** 4,534 ward-category pairs
* **Columns:** 11 aggregate metrics (including `computation_timestamp`, `has_sufficient_data`)

**Key Findings:**
* Average rejection rate: 16.3% across all ward-categories
* Average boilerplate rate: 65.7% (opportunity for improvement)
* Total open complaints (30d): 7,195 unresolved

---

### 04 - NLP Feature Engineering: Staff Remarks Analysis
**Objective:** Extract semantic features from staff remarks using NLP techniques

**NLP Pipeline:**
1. **Urgency Score:** Keyword-based scoring (35+ keywords)
   * High urgency: "urgent", "emergency", "immediate", "critical", "danger"
   * Health/safety: "overflow", "dengue", "disease", "health", "sewage", "stagnant"
   * Infrastructure: "collapse", "crack", "pothole", "fallen", "uprooted"
2. **TF-IDF Vectorization:**
   * Max features: 5,000 terms
   * Min document frequency: 5
   * Optimized sampling: Focus on non-boilerplate remarks (~137K diverse text)
   * Text preprocessing: lowercase, remove special chars, normalize whitespace
3. **Dimensionality Reduction:** Truncated SVD → 50 components
   * Compression: 5,000 TF-IDF features → 50 SVD components
   * Batch transformation: All 766K complaints using `mapInPandas` for distributed processing

**Output:**
* **Table:** `civic_lens.silver.bbmp_nlp_features`
* **Records:** 766,648 (100% coverage)
* **Columns:** 57 total
  * 1 urgency_score
  * 50 TF-IDF/SVD components (tfidf_feat_1 through tfidf_feat_50)
  * 6 metadata (complaint_id, category, ward_name_normalized, grievance_date, remark_is_boilerplate, remark_length)

**Optimization:**
* Spark-native batch processing (no Pandas conversion bottleneck)
* Smart sampling strategy for TF-IDF fitting (non-boilerplate heavy)
* Distributed transformation using `mapInPandas` for scalability

---

### 05 - Gold Layer: Unified ML & Analytics Table
**Objective:** Create denormalized, ML-ready table combining all features

**Join Strategy:**
1. **Base:** Silver complaints (766,648 rows)
2. **+NLP Features:** LEFT JOIN on `complaint_id` (100% match, 0 nulls)
3. **+Ward Aggregates:** LEFT JOIN on `(ward_name_normalized, category)` (~99% match)
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
* **Columns:** ~86 features
  * 19 core complaint fields
  * 51 NLP features (urgency + 50 TF-IDF/SVD)
  * 8 ward-category aggregates
  * 8 derived quality flags

**Data Quality:**
* ✅ Zero nulls in critical ML features
* ✅ Perfect join coverage (100%) for NLP features
* ✅ ~99% join coverage for ward-category aggregates
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
  * 50 TF-IDF/SVD semantic features
  * 8 boolean flags
* **Split:** 80/20 train-test (stratified)
  * Train: 593,786 samples
  * Test: 148,430 samples

**Models Trained:**

#### 1. XGBoost Multi-Class Classifier (Champion ⭐)
**Hyperparameters:**
* `objective`: multi:softprob
* `num_class`: 3
* `max_depth`: 6
* `learning_rate`: 0.1
* `n_estimators`: 100
* `subsample`: 0.8
* `colsample_bytree`: 0.8
* `eval_metric`: mlogloss
* `sample_weight`: Computed for class imbalance handling

**Performance (Test Set):**
* **Test Accuracy:** 97.42%
* **Train Accuracy:** 97.62%
* **Macro F1:** 0.8273
* **Weighted F1:** Not explicitly logged
* **Class Imbalance Handling:** Sample weights computed using `compute_sample_weight`

#### 2. Multinomial Logistic Regression (Baseline)
**Hyperparameters:**
* `multi_class`: multinomial
* `solver`: lbfgs
* `max_iter`: 1000

**Performance:**
* **Test Accuracy:** 95.17%
* **Macro F1:** 0.7613
* **Improvement:** XGBoost outperforms by **2.25%** in accuracy

**Model Artifacts:**
* MLflow experiment: `/Users/pawanvirat32@gmail.com/civic-lens/bbmp-outcome-prediction`
* XGBoost run ID: `6f5e49ac76664602b984b7c27aa40950`
* LogReg run ID: `e00f5e154cb44a56a230b4db3fbdeb8c`
* Artifacts logged: Model, confusion matrices, classification reports, feature importances

---

### 07 - Ward Risk Scoring: Complaint Outcome Predictions
**Objective:** Apply trained model to score all complaints and aggregate by ward-category

**Batch Inference Pipeline:**
1. **Load Model:** MLflow pyfunc from latest run (XGBoost)
2. **Prepare Features:** 71-column feature vector for 766,648 complaints (nulls filled with 0)
3. **Model Serialization:** Serialize model and broadcast to workers for distributed inference
4. **Predict Outcomes:** Score all complaints using `mapInPandas` for distributed processing
   * Returns: Hard class prediction + probability distribution [p_0, p_1, p_2]
5. **Aggregate to Ward-Category:** Group predictions by ward × category
6. **Compute Risk Scores:**
   * **Key Innovation:** Uses **average rejection probability** (soft predictions) instead of hard class counts
   * This avoids the "zero-wall" problem caused by class imbalance (Class 2 predictions are rare)
   * `rejection_risk_score`: Average of `prob_rejection` across all complaints in ward-category
   * `boilerplate_risk_score`: % boilerplate remarks
7. **Assign Risk Tiers:**
   * **High Risk:** rejection_risk_score > 10%
   * **Medium Risk:** 5% < rejection_risk_score ≤ 10%
   * **Low Risk:** rejection_risk_score ≤ 5%

**Output:**
* **Table:** `civic_lens.output.bangalore_ward_risk`
* **Records:** 4,534 ward-category combinations
* **Columns:** 10
  * `ward_name_normalized`, `category`
  * `total_complaints`
  * `predicted_resolved`, `predicted_closed`, `predicted_rejected` (counts)
  * `rejection_risk_score`, `boilerplate_risk_score` (percentages)
  * `risk_tier` (High/Medium/Low)
  * `scoring_timestamp`

**Risk Distribution:**
| Risk Tier | Combinations | Avg Rejection % |
|-----------|--------------|-----------------|
| **High** | 933 (20.6%) | 26.4% |
| **Medium** | 222 (4.9%) | 7.5% |
| **Low** | 3,379 (74.5%) | 1.8% |

**Overall Statistics:**
* **Total Combinations:** 4,534
* **Total Complaints Scored:** 766,648
* **Unique Wards:** 199
* **Unique Categories:** 32
* **Average Rejection Risk:** 6.3% (weighted average)
* **Average Boilerplate Rate:** 65.7%

**Technical Note:**
* Sparse-cell analysis shows median cell size of 13 complaints, with 25% of cells having ≤3 complaints
* Probabilistic scoring is critical for small cells where hard predictions would always be Class 1 (Closed)

---

## 🎯 Key Achievements & Resume Metrics

### Technical Achievements
* ✅ **Processed 766,648 civic complaints** across 6 years (2020-2025)
* ✅ **Built end-to-end ML pipeline** using Databricks lakehouse architecture
* ✅ **Achieved 97.42% prediction accuracy** with XGBoost multi-class classifier
* ✅ **Engineered ~86 features** including 50-dimensional TF-IDF/SVD semantic embeddings
* ✅ **Optimized Spark-native NLP** pipeline processing 766K documents using distributed `mapInPandas`
* ✅ **Zero data loss** through pipeline (100% join coverage for NLP features)
* ✅ **Created 4,534 ward-category risk scores** using probabilistic soft predictions

### Data Engineering Metrics
* **3-layer medallion architecture:** Bronze → Silver → Gold
* **6 Delta Lake tables** with auto-optimization and partitioning
* **100% data quality:** Zero nulls in critical features, validated schemas
* **Serverless compute:** Optimized for Databricks serverless Spark
* **Scalable NLP:** TF-IDF + SVD dimensionality reduction (5K → 50 features)
* **Distributed processing:** Used `mapInPandas` for batch transformations

### Machine Learning Metrics
* **Model:** XGBoost multi-class classifier (3 outcomes)
* **Accuracy:** 97.42% test, 97.62% train (minimal overfitting)
* **Class Imbalance Handling:** Stratified split, sample weighting
* **Probabilistic Risk Scoring:** Uses soft predictions to avoid sparse-cell bias
* **MLflow Integration:** Full experiment tracking, model versioning, artifact logging
* **Batch Inference:** Distributed scoring via serialized model + `mapInPandas` (766K predictions)

### Business Impact
* **Identified 933 high-risk areas** (ward-category pairs with >10% rejection probability)
* **Top 20.6% of ward-categories** flagged for immediate attention
* **65.7% boilerplate rate** uncovered (opportunity for service quality improvement)
* **Enabled proactive intervention** through 3-tier risk classification
* **Probabilistic scoring** handles sparse data better than hard predictions

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
  * TF-IDF/SVD embeddings + urgency scores
  
* **`civic_lens.silver.bbmp_ward_category_agg`**
  * 4,534 ward-category pairs, 11 columns
  * Aggregated service quality metrics

### Gold Layer
* **`civic_lens.gold.bbmp_complaints_enriched`**
  * 766,648 records, ~86 columns
  * ML-ready unified table
  * Perfect join coverage for NLP (100%), ~99% for aggregates

### Output Layer
* **`civic_lens.output.bangalore_ward_risk`**
  * 4,534 ward-category scores, 10 columns
  * Risk tiers (High/Medium/Low)
  * Probabilistic risk scores
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
| **Geospatial** | Custom geo_utils module |
| **Data Catalog** | Unity Catalog |
| **Orchestration** | Databricks Workflows |

---

## 📈 Model Performance Summary

| Metric | XGBoost (Champion) | Logistic Regression |
|--------|-------------------|---------------------|
| **Test Accuracy** | 97.42% | 95.17% |
| **Train Accuracy** | 97.62% | Not logged |
| **Macro F1** | 0.8273 | 0.7613 |
| **Training Time** | ~2-3 min | ~3-4 min |
| **Inference (766K)** | ~12 min (distributed) | ~10 min (distributed) |
| **Class Imbalance** | Sample weighting | Default |

**Winner:** XGBoost (+2.25% accuracy improvement)

**Implementation Notes:**
* Both models use stratified train-test split (80/20)
* XGBoost uses sample weighting to handle class imbalance
* Probabilistic predictions used for risk scoring to avoid sparse-cell bias
* Distributed inference via serialized model + `mapInPandas`

---

## 🚀 Deployment & Operationalization

### Model Deployment
* **Registry:** MLflow experiment tracking
* **Serving:** Batch inference via distributed `mapInPandas` with serialized model
* **Refresh:** Scheduled weekly scoring run
* **Monitoring:** Accuracy metrics, prediction distribution, risk tier counts

### Output Tables
* **Primary:** `civic_lens.output.bangalore_ward_risk`
* **Refresh:** Weekly full refresh (overwrite mode)
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
| **Join Coverage (NLP)** | 100% (perfect 1:1 join) |
| **Join Coverage (Agg)** | ~99% (rare combinations may miss) |
| **Schema Validation** | ✅ All 7 stages validated |
| **Outcome Label Quality** | 96.8% labeled (3.2% in-progress) |
| **Temporal Consistency** | ✅ 2020-2025 continuous |
| **Duplicate Detection** | 0 duplicates (unique `complaint_id`) |
| **Feature Coverage** | ~86/86 features populated |

---

## 🎓 Key Learnings & Best Practices

### Data Engineering
* **Medallion architecture** (Bronze/Silver/Gold) provides clear data lineage
* **Delta Lake optimization** (auto-compact, optimize-write) critical for performance
* **Spark-native processing** avoids Pandas bottlenecks (766K rows, distributed transforms)
* **Custom JSON parsing** required for non-standard format (`{fields: [...], records: [...]}`)
* **Geospatial utils** centralized in `geo_utils.py` for ward name normalization

### Machine Learning
* **Class imbalance** handled via stratified split + sample weighting
* **Probabilistic risk scoring** critical for sparse cells (avoids "zero-wall" from hard predictions)
* **Feature engineering** includes domain-specific flags (`is_problem_area`, `is_high_boilerplate_area`)
* **Ensemble methods** (XGBoost) outperform linear models significantly
* **MLflow tracking** essential for experiment reproducibility
* **Distributed inference** via `mapInPandas` with serialized model enables scalability

### NLP
* **TF-IDF + SVD** effective for large-scale text feature extraction
* **Smart sampling** for TF-IDF fitting (focus on non-boilerplate) improves signal
* **Domain keywords** (Indian context: "dengue", "sewage", "overflow") boost urgency detection
* **Boilerplate detection** (88% rate) reveals automation opportunity
* **Batch transformation** via `mapInPandas` scales to 766K documents

---

## 📝 Future Enhancements

### Short-Term (v2.0)
* [ ] Real-time inference endpoint for live complaint scoring
* [ ] Dashboard with ward-level heat maps and trend analysis
* [ ] Automated Slack/email alerts for high-risk complaints
* [ ] A/B test intervention strategies (staff training, routing)

### Medium-Term (v2.5)
* [ ] Temporal analysis: Time-series forecasting of complaint volumes
* [ ] Geospatial analysis: Spatial clustering of high-risk areas using ward polygons
* [ ] Root cause analysis: SHAP values for feature attribution
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
Databricks Lakehouse Platform. 766K complaints, 97.42% classification accuracy.
```

---

**Last Updated:** June 2026  
**Pipeline Version:** 1.0  
**Status:** ✅ Production-Ready

---
