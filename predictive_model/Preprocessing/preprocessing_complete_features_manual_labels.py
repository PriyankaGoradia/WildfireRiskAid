import pandas as pd
import json

# Load the CSV
df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Data/Alberta_Complete_Wildfire_Features.csv")

# Extract coordinates from .geo
df['coordinates'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'])
df[['Lon', 'Lat']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)

# Drop unused columns
df = df.drop(columns=['.geo', 'system:index', 'coordinates'])

# Add fire label based on rule (can be replaced with MODIS label later)
df['Fire_Label'] = ((df['NDVI'] < 0.2) & (df['NBR'] < 0.3)).astype(int)

# Reorder columns
cols = ['Lat', 'Lon', 'NDVI', 'NBR', 'NDWI', 'Temp', 'Wind_Dir', 'Wind_Spd',
        'Humidity', 'Elev', 'Slope', 'Fire_Label']
df = df[cols]

# Save cleaned dataset
df.to_csv("preprocessed_fire_data.csv", index=False)
print("Preprocessing complete. File saved as preprocessed_fire_data.csv")
