import pandas as pd

# Load MODIS fire detection CSV
df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Data-2024/MODIS_Alberta_2024.csv")

# Keep only relevant columns
df = df[[
    'latitude',
    'longitude',
    'acq_date',
    'acq_time',
    'brightness',
    'confidence',
    'satellite',
    'instrument',
    'daynight'
]]

# Rename for clarity
df = df.rename(columns={
    'latitude': 'Lat',
    'longitude': 'Lon',
    'acq_date': 'Date',
    'acq_time': 'Time',
    'brightness': 'Brightness',
    'confidence': 'Confidence',
    'satellite': 'Satellite',
    'instrument': 'Instrument',
    'daynight': 'DayNight'
})

# Convert date and time to a single datetime column
df['Timestamp'] = pd.to_datetime(df['Date'] + df['Time'].astype(str).str.zfill(4), format='%Y-%m-%d%H%M')

# Optional: filter by confidence threshold (e.g., keep only high confidence)
df = df[df['Confidence'] >= 60]

# Drop original date/time columns
df = df.drop(columns=['Date', 'Time'])

# Reset index
df = df.reset_index(drop=True)

# Save cleaned file
df.to_csv("cleaned_MODIS_Alberta_2024.csv", index=False)
print("MODIS fire data cleaned and saved as cleaned_modis_fires.csv")
