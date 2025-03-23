import pandas as pd
import json

# Load the raw dataset
df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Data/Alberta_Complete_Wildfire_Features.csv")

# Show column names to verify structure
print("Available columns:", df.columns.tolist())

# Parse coordinates from .geo
df['coordinates'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'])
df[['Lon', 'Lat']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)

# Check and convert timestamp if present
if 'system:time_start' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['system:time_start'], unit='ms')
    df = df.drop(columns=['system:time_start'])
else:
    print("Warning: 'system:time_start' not found. Timestamp will be skipped.")
    df['Timestamp'] = pd.NaT  # Fill with empty timestamps if needed

# Drop unneeded columns
df = df.drop(columns=['.geo', 'system:index', 'coordinates'], errors='ignore')

# Reorder columns (adjust to what’s present)
expected_cols = ['Lat', 'Lon', 'NDVI', 'NBR', 'NDWI', 'Temp', 'Wind_Dir', 'Wind_Spd',
                 'Humidity', 'Elev', 'Slope', 'Timestamp']
df = df[[col for col in expected_cols if col in df.columns]]

# Save to CSV
df.to_csv("preprocessed_fire_data.csv", index=False)
print("✅ Preprocessed feature data saved as preprocessed_fire_data.csv")
