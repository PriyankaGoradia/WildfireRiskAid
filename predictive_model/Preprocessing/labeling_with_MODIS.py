import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import timedelta

# Load datasets
gee_df = pd.read_csv("preprocessed_fire_data.csv")
modis_df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Data/cleaned_modis_fires.csv")

# Convert to datetime
modis_df['Timestamp'] = pd.to_datetime(modis_df['Timestamp'])
# If GEE dataset doesn't have Timestamp yet, add a dummy one here for structure
if 'Timestamp' not in gee_df.columns:
    raise ValueError("Timestamp column not found in GEE data!")

gee_df['Timestamp'] = pd.to_datetime(gee_df['Timestamp'])

# Convert to GeoDataFrames
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

# Spatial join — buffer GEE points to simulate proximity (~1km ≈ 0.01 deg)
gee_gdf['geometry'] = gee_gdf.buffer(0.01)

# Join MODIS fires that fall within the buffered GEE areas
joined = gpd.sjoin(gee_gdf, modis_gdf, how='left', predicate='contains')

# Time-based matching within ±1 day
def match_fire(row):
    fire_time = row['Timestamp_right']
    gee_time = row['Timestamp_left']
    if pd.isna(fire_time):
        return 0
    return 1 if abs((gee_time - fire_time).days) <= 1 else 0

# Create Fire_Label column
joined['Fire_Label'] = joined.apply(match_fire, axis=1)

# Drop duplicates and clean up
result = joined.drop_duplicates(subset='Timestamp_left')
result = result.rename(columns={'Timestamp_left': 'Timestamp'})
result = result.drop(columns=['geometry', 'Timestamp_right'])

# Final clean-up: select only original GEE + Fire_Label
final_cols = [col for col in gee_df.columns if col != 'Fire_Label'] + ['Fire_Label']
result = result[final_cols]

# Save labeled dataset
result.to_csv("preprocessed_with_real_fire_labels.csv", index=False)
print("Fire labels assigned using MODIS data and saved as preprocessed_with_real_fire_labels.csv")
