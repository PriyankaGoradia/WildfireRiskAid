import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load datasets
gee_df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Data-2023/preprocessed_Alberta_Complete_Wildfire_Feature.csv")
modis_df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Data-2023/cleaned_MODIS_fires.csv")

# Create GeoDataFrames with WGS84 (Lat/Lon)
gee_gdf = gpd.GeoDataFrame(
    gee_df,
    geometry=gpd.points_from_xy(gee_df['Lon'], gee_df['Lat']),
    crs="EPSG:4326"
)
modis_gdf = gpd.GeoDataFrame(
    modis_df,
    geometry=gpd.points_from_xy(modis_df['Lon'], modis_df['Lat']),
    crs="EPSG:4326"
)

# Save original point geometry for lat/lon extraction later
gee_gdf['point_geometry'] = gee_gdf.geometry

# Reproject to a metric CRS (EPSG:3857) for proper buffering
gee_gdf = gee_gdf.to_crs(epsg=3857)
modis_gdf = modis_gdf.to_crs(epsg=3857)

# Buffer GEE points ~1km
gee_gdf['geometry'] = gee_gdf.geometry.buffer(1000)

# Spatial join to check which MODIS fire points fall inside the buffer
joined = gpd.sjoin(gee_gdf, modis_gdf, how='left', predicate='contains')

# Assign label: 1 = fire detected nearby, 0 = no fire nearby
joined['Fire_Label'] = joined['index_right'].notna().astype(int)

# Restore original point geometry and project back to WGS84
joined = joined.set_geometry('point_geometry').to_crs(epsg=4326)
joined['Lat'] = joined.geometry.y
joined['Lon'] = joined.geometry.x

# Drop unnecessary columns
joined = joined.drop(columns=['geometry', 'point_geometry', 'index_right'])

# Safely build final column list
available_cols = list(joined.columns)
original_cols = [col for col in gee_df.columns if col in available_cols and col != 'Fire_Label']
final_cols = original_cols + ['Fire_Label']
result = joined[final_cols]


# Save to CSV
result.to_csv("preprocessed_with_real_fire_labels.csv", index=False)
print("✅ Saved: preprocessed_with_real_fire_labels.csv with real fire labels (no timestamp)")
