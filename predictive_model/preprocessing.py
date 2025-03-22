import pandas as pd
import ast

# Load the exported Earth Engine CSV
df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Alberta_NDVI_NBR_Extracted.csv")

# Convert the .geo column from string to actual coordinates
df['coordinates'] = df['.geo'].apply(lambda x: ast.literal_eval(x)['coordinates'])

# Extract lon and lat into separate columns
df[['lon', 'lat']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)

# Drop unnecessary columns
df = df.drop(columns=['system:index', '.geo', 'coordinates'])

# Create a fire risk label: simulate with thresholds on NDVI & NBR
df['Fire_Label'] = ((df['NDVI'] < 0.2) & (df['NBR'] < 0.3)).astype(int)

# Save the preprocessed CSV
df.to_csv("preprocessed_data.csv", index=False)

print("Preprocessing complete. File saved as preprocessed_data.csv")
