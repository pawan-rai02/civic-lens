"""
Shared GeoJSON helper functions for civic-lens project.

Purpose: Inspect and join GeoJSON ward/borough boundaries against complaint data.
Handles the reality that GeoJSON property names vary (boro_name vs BoroName, etc.).
"""

import pandas as pd
from typing import Dict, List, Tuple


def inspect_geojson(path: str) -> None:
    """
    Load a GeoJSON file and print its structure.
    
    Prints:
    - All property columns
    - First feature's properties
    - Sample property values
    
    Use this to find the exact property key holding ward/borough names.
    
    Args:
        path: Path to .geojson file (S3 path or local)
    
    Example:
        inspect_geojson("s3://civiclens-data/geojson/nyc_boroughs.geojson")
    """
    import geopandas as gpd  # Lazy import
    print(f"Loading: {path}")
    gdf = gpd.read_file(path)
    
    print(f"\n📊 Total features: {len(gdf)}")
    print(f"\n📋 Columns:")
    for col in gdf.columns:
        print(f"  - {col}")
    
    print(f"\n🔍 First feature properties:")
    first_row = gdf.iloc[0]
    for col in gdf.columns:
        if col != 'geometry':
            print(f"  {col}: {first_row[col]}")
    
    print(f"\n📝 Sample values for key columns:")
    # Try common name columns
    name_candidates = [col for col in gdf.columns if 'name' in col.lower() or 'boro' in col.lower() or 'ward' in col.lower()]
    for col in name_candidates:
        print(f"\n  {col}:")
        print(f"    Unique values: {gdf[col].nunique()}")
        print(f"    Sample: {gdf[col].head(5).tolist()}")


def normalize_borough(name: str) -> str:
    """
    Normalize NYC borough names for GeoJSON joins.
    
    Steps:
    - Handle null/empty values
    - Strip whitespace and lowercase
    - Filter out UNSPECIFIED
    - Handle common variations (THE BRONX -> bronx, etc.)
    
    Args:
        name: Raw borough name from 311 data
        
    Returns:
        Normalized borough name for GeoJSON matching, or None if invalid
        
    Example:
        normalize_borough("MANHATTAN") -> "manhattan"
        normalize_borough("THE BRONX") -> "bronx"
        normalize_borough("UNSPECIFIED") -> None
    """
    if pd.isna(name) or name is None or str(name).strip() == "":
        return None
    
    name = str(name).strip().upper()
    
    # Filter out invalid values
    if name in ["UNSPECIFIED", "UNKNOWN", "N/A", ""]:
        return None
    
    # Normalize common variations
    name = name.replace("THE ", "")  # "THE BRONX" -> "BRONX"
    name = name.strip().lower()
    
    return name


def normalize_ward_name(name: str) -> str:
    """
    Normalize ward/borough names for matching.
    
    Steps:
    - Strip whitespace
    - Lowercase
    - Remove the literal word "Ward" (common in Bangalore data)
    
    Args:
        name: Raw ward/borough name
        
    Returns:
        Normalized name for matching
        
    Example:
        normalize_ward_name("  Jnanabharathi Ward  ") -> "jnanabharathi"
    """
    if pd.isna(name) or name is None:
        return ""
    
    name = str(name).strip().lower()
    # Remove common suffixes
    name = name.replace(" ward", "").replace("-ward", "")
    return name.strip()


def check_join_match_rate(
    data_names: List[str], 
    geo_names: List[str],
    normalize: bool = True
) -> Tuple[float, List[str]]:
    """
    Check what percentage of data names have matches in GeoJSON.
    
    Args:
        data_names: List of ward/borough names from your complaint data
        geo_names: List of ward/borough names from GeoJSON properties
        normalize: Whether to normalize names before matching
        
    Returns:
        (match_rate, unmatched_list)
        - match_rate: Percentage (0-100) of data names that found a match
        - unmatched_list: List of data names that didn't match
        
    Example:
        >>> data_names = ["MANHATTAN", "BROOKLYN", "BRONX", "UNKNOWN"]
        >>> geo_names = ["Manhattan", "Brooklyn", "Bronx", "Queens", "Staten Island"]
        >>> rate, unmatched = check_join_match_rate(data_names, geo_names)
        >>> print(f"Match rate: {rate}%")
        >>> print(f"Unmatched: {unmatched}")
    """
    # Convert to sets for faster lookup
    if normalize:
        data_set = {normalize_ward_name(n) for n in data_names if n}
        geo_set = {normalize_ward_name(n) for n in geo_names if n}
    else:
        data_set = {str(n).strip() for n in data_names if n}
        geo_set = {str(n).strip() for n in geo_names if n}
    
    # Find matches
    matched = data_set & geo_set
    unmatched = sorted(data_set - geo_set)
    
    # Calculate rate
    match_rate = (len(matched) / len(data_set) * 100) if data_set else 0
    
    return match_rate, unmatched


def build_manual_mapping_dict() -> Dict[str, str]:
    """
    Manual mapping for ward/borough names that don't match automatically.
    
    Fill this in after running check_join_match_rate() to see unmatched names.
    
    Returns:
        Dictionary mapping data names -> GeoJSON names
        
    Example:
        {
            "Jnanabharathi Ward": "Jnanabharathi",
            "HSR Layout": "HSR Layout Ward",
            "Whitefield": "Whitefield Ward"
        }
    """
    # NYC manual mappings (if any)
    nyc_mappings = {
        # Add any NYC borough name variations here
        # e.g., "MN": "MANHATTAN"
    }
    
    # Bangalore manual mappings (if any)
    bangalore_mappings = {
        # Add Bangalore ward name variations here after inspection
        # e.g., "Jnanabharathi Ward": "Jnanabharathi"
    }
    
    return {**nyc_mappings, **bangalore_mappings}


def apply_manual_mapping(name: str, mapping_dict: Dict[str, str]) -> str:
    """
    Apply manual mapping to a name, falling back to original if not found.
    
    Args:
        name: Original name
        mapping_dict: Manual mapping dictionary
        
    Returns:
        Mapped name or original if no mapping exists
    """
    return mapping_dict.get(name, name)
