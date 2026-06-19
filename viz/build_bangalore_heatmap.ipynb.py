# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Bangalore Complaint Risk Heatmap Generator
# MAGIC %md
# MAGIC # Bangalore Complaint Risk Heatmap Generator
# MAGIC
# MAGIC **Purpose:** Generate an interactive choropleth heatmap of Bangalore ward-level complaint risk scores.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC - Load `civic_lens.ml.bangalore_ward_scores` (ward × complaint_type risk scores)
# MAGIC - Aggregate to ward-level weighted risk scores (by rejection_risk_score)
# MAGIC - Join with Bangalore ward GeoJSON boundaries using geo_utils normalization
# MAGIC - **Ward name matching validation** (known risk point - explicit match-rate check)
# MAGIC - Render Folium choropleth with interactive popups
# MAGIC
# MAGIC **Output:** `viz/output/bangalore_heatmap.html` (interactive map)
# MAGIC
# MAGIC **Data Flow:**
# MAGIC ```
# MAGIC civic_lens.ml.bangalore_ward_scores
# MAGIC     ↓
# MAGIC Aggregate by ward (weighted by complaint_count)
# MAGIC     ↓
# MAGIC geo_utils.normalize_ward() - standardize ward names
# MAGIC     ↓
# MAGIC Join with bangalore_wards.geojson (match rate check)
# MAGIC     ↓
# MAGIC Folium choropleth visualization
# MAGIC     ↓
# MAGIC bangalore_heatmap.html
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Install required libraries
# MAGIC %pip install geopandas folium -q

# COMMAND ----------

# DBTITLE 1,Restart Python kernel
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import libraries and setup
import sys
import os
import pandas as pd
import geopandas as gpd
import folium
from pyspark.sql import SparkSession

# Add src directory to path for geo_utils
sys.path.insert(0, '/Workspace/Users/pawanvirat32@gmail.com/civic-lens/src')
from geo_utils import normalize_ward_name

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

print("✓ Libraries loaded successfully")
print("✓ geo_utils.normalize_ward_name imported for ward name matching")

# COMMAND ----------

# DBTITLE 1,Load ward risk data from Databricks
print("=" * 70)
print("BANGALORE COMPLAINT RISK HEATMAP GENERATOR")
print("=" * 70)
print("\n[1/6] Loading ward risk data from Databricks...")

# Load the ward risk scores from civic_lens.output.bangalore_ward_risk
risk_df = spark.table("civic_lens.output.bangalore_ward_risk").toPandas()

# Rename columns to match expected schema
risk_df = risk_df.rename(columns={
    'ward_name_normalized': 'ward_name',
    'category': 'complaint_type'
})

print(f"   ✓ Loaded {len(risk_df):,} ward × complaint_type combinations")
print(f"   ✓ Columns: {list(risk_df.columns)}")
print(f"   ✓ Unique wards: {risk_df['ward_name'].nunique()}")
print(f"   ✓ Ward names are pre-normalized")

# Display sample
display(risk_df.head())

# COMMAND ----------

# DBTITLE 1,Aggregate risk scores by ward
print("\n[2/6] Aggregating risk scores by ward...")

# Use rejection_risk_score as primary metric (blended deflection score)
# Weight by total_complaints to get ward-level aggregate
ward_agg = risk_df.groupby('ward_name').apply(
    lambda x: pd.Series({
        'total_complaints': x['total_complaints'].sum(),
        'weighted_risk_score': (x['rejection_risk_score'] * x['total_complaints']).sum() / x['total_complaints'].sum() if x['total_complaints'].sum() > 0 else 0,
        'avg_rejections': x['predicted_rejections'].sum() / x['total_complaints'].sum() if x['total_complaints'].sum() > 0 else 0,
        'risk_tier': x['risk_tier'].mode()[0] if not x['risk_tier'].empty else 'LOW'
    })
).reset_index()

# Get top 5 complaint types per ward
top_complaints_by_ward = {}
for ward in risk_df['ward_name'].unique():
    ward_data = risk_df[risk_df['ward_name'] == ward].nlargest(5, 'total_complaints')
    top_complaints_by_ward[ward] = ward_data[['complaint_type', 'total_complaints', 'rejection_risk_score']].to_dict('records')

ward_agg['top_complaints'] = ward_agg['ward_name'].map(top_complaints_by_ward)

print(f"   ✓ Aggregated to {len(ward_agg)} wards")
print(f"\n   Ward Risk Summary (Top 10 by risk):")
for _, row in ward_agg.sort_values('weighted_risk_score', ascending=False).head(10).iterrows():
    print(f"      {row['ward_name']:30} | Score: {row['weighted_risk_score']:.3f} | Tier: {row['risk_tier']:6} | {row['total_complaints']:,} complaints")

display(ward_agg.head(10))

# COMMAND ----------

# DBTITLE 1,Load Bangalore ward GeoJSON boundaries
print("\n[3/6] Loading Bangalore ward GeoJSON boundaries...")

geojson_path = '/Workspace/Users/pawanvirat32@gmail.com/civic-lens/src/tmp_bangalore_wards.geojson'  # Verify this path exists or update accordingly
gdf = gpd.read_file(geojson_path)

print(f"   ✓ Loaded {len(gdf)} ward geometries")
print(f"   ✓ GeoJSON columns: {list(gdf.columns)}")

# Identify the ward name column in GeoJSON (might be 'ward_name', 'WARD_NAME', 'name', etc.)
ward_col_candidates = [c for c in gdf.columns if 'ward' in c.lower() or 'name' in c.lower()]
print(f"   ℹ️  Potential ward name columns: {ward_col_candidates}")

display(gdf.head())

# COMMAND ----------

# DBTITLE 1,Ward name matching validation using geo_utils
print("\n[4/6] Ward name matching validation (using geo_utils)...")
print("   ⚠️  This is a known risk point - checking match rate explicitly\n")

# Identify the ward name column in GeoJSON
# Common patterns: 'ward_name', 'WARD_NAME', 'name', 'Ward_Name', 'KGISWardName'
ward_col = None
for candidate in ['ward_name', 'WARD_NAME', 'Ward_Name', 'name', 'NAME', 'KGISWardName', 'WardName']:
    if candidate in gdf.columns:
        ward_col = candidate
        break

if ward_col is None:
    print(f"   ❌ Could not find ward name column in GeoJSON. Available columns: {list(gdf.columns)}")
    print("   Please specify the correct column manually.")
else:
    print(f"   ✓ Using '{ward_col}' as ward name column in GeoJSON\n")

# Apply geo_utils normalization to both datasets
print("   Normalizing ward names with geo_utils.normalize_ward_name()...")
ward_agg['ward_norm'] = ward_agg['ward_name'].apply(normalize_ward_name)
gdf['ward_norm'] = gdf[ward_col].apply(normalize_ward_name)

print(f"   ✓ Normalization applied\n")

# Check match rate BEFORE merging
risk_wards = set(ward_agg['ward_norm'])
geo_wards = set(gdf['ward_norm'])

matched = risk_wards & geo_wards
unmatched_risk = risk_wards - geo_wards
unmatched_geo = geo_wards - risk_wards

match_rate = len(matched) / len(risk_wards) * 100 if len(risk_wards) > 0 else 0

print(f"   MATCH RATE ANALYSIS:")
print(f"   " + "="*60)
print(f"   Risk data wards:     {len(risk_wards)}")
print(f"   GeoJSON wards:       {len(geo_wards)}")
print(f"   Matched:             {len(matched)} ({match_rate:.1f}%)")
print(f"   Unmatched (risk):    {len(unmatched_risk)}")
print(f"   Unmatched (geo):     {len(unmatched_geo)}")
print(f"   " + "="*60)

if len(unmatched_risk) > 0:
    print(f"\n   ⚠️  Unmatched wards in risk data (will not appear on map):")
    for ward in sorted(list(unmatched_risk))[:10]:  # Show first 10
        original = ward_agg[ward_agg['ward_norm'] == ward]['ward_name'].iloc[0]
        print(f"      • {original} → (normalized: {ward})")
    if len(unmatched_risk) > 10:
        print(f"      ... and {len(unmatched_risk) - 10} more")

if len(unmatched_geo) > 0:
    print(f"\n   ℹ️  Unmatched wards in GeoJSON (no risk data):")
    for ward in sorted(list(unmatched_geo))[:10]:  # Show first 10
        original = gdf[gdf['ward_norm'] == ward][ward_col].iloc[0]
        print(f"      • {original} → (normalized: {ward})")
    if len(unmatched_geo) > 10:
        print(f"      ... and {len(unmatched_geo) - 10} more")

if match_rate < 80:
    print(f"\n   ❌ WARNING: Match rate is low ({match_rate:.1f}%). Map will be incomplete.")
    print(f"      Consider reviewing geo_utils.normalize_ward_name() logic.")
elif match_rate < 95:
    print(f"\n   ⚠️  Match rate is acceptable ({match_rate:.1f}%) but could be improved.")
else:
    print(f"\n   ✅ Match rate is good ({match_rate:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Merge risk data with GeoJSON
print("\n[5/6] Merging risk data with GeoJSON...")

# Merge on normalized ward names
gdf_merged = gdf.merge(
    ward_agg,
    on='ward_norm',
    how='left'
)

# Check merge success
matched = gdf_merged['weighted_risk_score'].notna().sum()
print(f"   ✓ Merged {matched}/{len(gdf)} wards successfully")

if matched < len(gdf):
    unmatched_list = gdf_merged[gdf_merged['weighted_risk_score'].isna()][ward_col].tolist()
    print(f"   ⚠️  {len(unmatched_list)} wards have no risk data: {unmatched_list[:5]}{'...' if len(unmatched_list) > 5 else ''}")

# Fill missing values with 0 for unmapped wards
gdf_merged['weighted_risk_score'] = gdf_merged['weighted_risk_score'].fillna(0)
gdf_merged['total_complaints'] = gdf_merged['total_complaints'].fillna(0).astype(int)
gdf_merged['risk_tier'] = gdf_merged['risk_tier'].fillna('LOW')
gdf_merged['avg_rejections'] = gdf_merged['avg_rejections'].fillna(0)

print(f"\n   Merged data preview:")
display(gdf_merged[[ward_col, 'weighted_risk_score', 'total_complaints', 'risk_tier', 'avg_rejections']].head(10))

# COMMAND ----------

# DBTITLE 1,Create Folium choropleth heatmap
print("\n[6/6] Creating Folium choropleth heatmap...")

# Calculate map center (Bangalore coordinates)
bangalore_center = [12.9716, 77.5946]  # Bangalore city center

# Create base map
m = folium.Map(
    location=bangalore_center,
    zoom_start=11,
    tiles='CartoDB positron'
)

# Create choropleth layer
folium.Choropleth(
    geo_data=gdf_merged.to_json(),
    name='Ward Risk',
    data=gdf_merged,
    columns=[ward_col, 'weighted_risk_score'],
    key_on=f'feature.properties.{ward_col}',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Rejection Risk Score (Weighted)',
    nan_fill_color='lightgray'
).add_to(m)

# Add interactive popups with detailed info
for _, row in gdf_merged.iterrows():
    # Build top complaints HTML
    top_complaints_html = "<br><b>Top 5 Complaint Types:</b><br>"
    top_complaints = row['top_complaints']
    if isinstance(top_complaints, list) and len(top_complaints) > 0:
        for i, complaint in enumerate(top_complaints, 1):
            complaint_type = complaint['complaint_type']
            count = int(complaint['total_complaints'])
            risk = complaint['rejection_risk_score'] if not pd.isna(complaint['rejection_risk_score']) else 0
            top_complaints_html += f"<span style='font-size:11px'>{i}. <b>{complaint_type}</b> ({count:,} complaints, risk: {risk:.2f})</span><br>"
    else:
        top_complaints_html += "<i>No data</i><br>"
    
    ward_display_name = row[ward_col] if pd.notna(row[ward_col]) else 'Unknown'
    risk_score = row['weighted_risk_score'] if pd.notna(row['weighted_risk_score']) else 0
    total_complaints = int(row['total_complaints']) if pd.notna(row['total_complaints']) else 0
    risk_tier = row['risk_tier'] if pd.notna(row['risk_tier']) else 'LOW'
    avg_rejections_val = row['avg_rejections'] if pd.notna(row['avg_rejections']) else 0
    
    folium.GeoJson(
        row.geometry,
        style_function=lambda x: {
            'fillColor': 'transparent',
            'color': 'transparent',
            'weight': 0
        },
        tooltip=folium.Tooltip(
            f"""
            <b>{ward_display_name}</b><br>
            <hr style='margin:4px 0'>
            Rejection Risk Score: <b>{risk_score:.3f}</b><br>
            Risk Tier: <b>{risk_tier}</b><br>
            Total Complaints: <b>{total_complaints:,}</b><br>
            Avg Rejection Rate: <b>{avg_rejections_val:.1%}</b><br>
            {top_complaints_html}
            """,
            style='font-size: 12px; font-family: Arial;'
        )
    ).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

print(f"   ✓ Choropleth layer created")
print(f"   ✓ Interactive popups added for {len(gdf_merged)} wards")

# COMMAND ----------

# DBTITLE 1,Save heatmap and display summary
# Save to output directory
output_path = '/Workspace/Users/pawanvirat32@gmail.com/civic-lens/viz/output/bangalore_heatmap.html'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
m.save(output_path)

print(f"\n   ✓ Heatmap saved to: {output_path}")
print(f"   ✓ Map centered at: {bangalore_center}")
print(f"   ✓ Zoom level: 11")

# Print summary statistics
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total wards: {len(gdf_merged)}")
print(f"Wards with risk data: {(gdf_merged['weighted_risk_score'] > 0).sum()}")
print(f"Total complaints analyzed: {int(gdf_merged['total_complaints'].sum()):,}")
print(f"Risk score range: {gdf_merged[gdf_merged['weighted_risk_score'] > 0]['weighted_risk_score'].min():.3f} - {gdf_merged['weighted_risk_score'].max():.3f}")
print(f"\nRisk tier distribution:")
for tier in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
    count = len(gdf_merged[gdf_merged['risk_tier'] == tier])
    if count > 0:
        print(f"  {tier}: {count} wards")
print("\n" + "=" * 70)
print("✅ Bangalore Complaint Risk Heatmap generation complete!")
print("=" * 70)

# Display the map inline
m

# COMMAND ----------


