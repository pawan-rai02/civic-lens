# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "geopandas",
# ]
# ///
# DBTITLE 1,GeoJSON Exploration & Verification
# MAGIC %md
# MAGIC # GeoJSON Exploration & Verification
# MAGIC
# MAGIC This notebook inspects the NYC and Bangalore GeoJSON boundary files to:
# MAGIC 1. Identify the exact property names for ward/borough identifiers
# MAGIC 2. Test the geo_utils.py helper functions
# MAGIC 3. Check match rates between complaint data and GeoJSON properties
# MAGIC
# MAGIC **GeoJSON Files:**
# MAGIC - NYC: `s3://civiclens-data/geojson/nyc_boroughs.geojson`
# MAGIC - Bangalore: `s3://civiclens-data/geojson/bangalore_wards.geojson`

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install geopandas

# COMMAND ----------

# DBTITLE 1,Import and add src to path
import sys
sys.path.append('/Workspace/Users/pawanvirat32@gmail.com/civic-lens/src')

from geo_utils import inspect_geojson, normalize_ward_name, check_join_match_rate, build_manual_mapping_dict
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Inspect NYC boroughs GeoJSON
print("=" * 60)
print("NYC BOROUGHS GEOJSON INSPECTION")
print("=" * 60)

# Download from S3 to workspace temp file first (geopandas can't read S3 directly in Databricks)
tmp_path = "/Workspace/Users/pawanvirat32@gmail.com/civic-lens/src/tmp_nyc_boroughs.geojson"
dbutils.fs.cp("s3://civiclens-data/geojson/nyc_boroughs.geojson", f"file:{tmp_path}")
inspect_geojson(tmp_path)

# COMMAND ----------

# DBTITLE 1,Inspect Bangalore wards GeoJSON
print("=" * 60)
print("BANGALORE WARDS GEOJSON INSPECTION")
print("=" * 60)

# Download from S3 to workspace temp file first (geopandas can't read S3 directly in Databricks)
tmp_path = "/Workspace/Users/pawanvirat32@gmail.com/civic-lens/src/tmp_bangalore_wards.geojson"
dbutils.fs.cp("s3://civiclens-data/geojson/bangalore_wards.geojson", f"file:{tmp_path}")
inspect_geojson(tmp_path)

# COMMAND ----------

# DBTITLE 1,Test normalization function
# Test ward name normalization
test_names = [
    "  Jnanabharathi Ward  ",
    "HSR Layout Ward",
    "MANHATTAN",
    "  Brooklyn  "
]

print("Testing normalize_ward_name():")
for name in test_names:
    normalized = normalize_ward_name(name)
    print(f"  '{name}' -> '{normalized}'")

# COMMAND ----------

# DBTITLE 1,Check NYC borough match rate
# Get unique borough names from silver table
df_nyc = spark.table("civic_lens.silver.nyc_311_cleaned")
data_boroughs = [row.borough for row in df_nyc.select("borough").distinct().collect()]

print(f"NYC Complaint data unique boroughs: {len(data_boroughs)}")
print(f"Boroughs: {sorted(data_boroughs)}")

# Load GeoJSON and extract actual borough names from BoroName column
import geopandas as gpd
tmp_path = "/Workspace/Users/pawanvirat32@gmail.com/civic-lens/src/tmp_nyc_boroughs.geojson"
gdf = gpd.read_file(tmp_path)
geo_boroughs = gdf['BoroName'].tolist()

print(f"\nGeoJSON borough names: {sorted(geo_boroughs)}")

match_rate, unmatched = check_join_match_rate(data_boroughs, geo_boroughs)
print(f"\nMatch rate: {match_rate:.1f}%")
print(f"Unmatched: {unmatched}")

# COMMAND ----------

# DBTITLE 1,Results summary
print("=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("1. Update geo_utils.py manual mapping dict with unmatched names")
print("2. Use these functions in 02_clean_silver notebooks to add clean ward/borough keys")
print("3. Use in viz/ scripts for choropleth joins")

# COMMAND ----------

# DBTITLE 1,Check Bangalore ward match rate
# MAGIC %md
# MAGIC # Summary of Findings
# MAGIC
# MAGIC ## ✅ NYC - Ready to Use
# MAGIC * **GeoJSON Property**: `BoroName`
# MAGIC * **Match Rate**: 83.3%
# MAGIC * **Unmatched**: Only "UNSPECIFIED" (expected for null/invalid data)
# MAGIC * **Status**: ✅ Normalization working perfectly - no manual mapping needed
# MAGIC
# MAGIC ## 🔍 Bangalore - Pending Silver Table
# MAGIC * **GeoJSON Property**: `KGISWardName` 
# MAGIC * **Total Wards in GeoJSON**: 243
# MAGIC * **Status**: Silver table not created yet - will check match rate after ingestion
# MAGIC
# MAGIC ## 📝 Implementation Plan
# MAGIC
# MAGIC ### For NYC (ready now):
# MAGIC ```python
# MAGIC # In 02_clean_silver_nyc.py, add:
# MAGIC from geo_utils import normalize_ward_name
# MAGIC
# MAGIC df = df.withColumn(
# MAGIC     "borough_normalized",
# MAGIC     F.when(F.col("borough") != "UNSPECIFIED", 
# MAGIC            F.lower(F.trim(F.col("borough"))))
# MAGIC     .otherwise(None)
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### For Bangalore (after silver table exists):
# MAGIC 1. Run similar match rate check against `civic_lens.silver.bangalore_complaints_cleaned`
# MAGIC 2. Add any required manual mappings to `geo_utils.py`
# MAGIC 3. Apply normalization in cleaning notebook
