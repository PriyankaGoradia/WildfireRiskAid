import pandas as pd
import ast

# Load data
df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Alberta_NDVI_NBR_Extracted.csv")

# Clean up geometry: convert '.geo' to lat/lon
df['coordinates'] = df['.geo'].apply(lambda x: ast.literal_eval(x)['coordinates'])
df[['lon', 'lat']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)

# Drop unnecessary columns
df = df.drop(columns=['system:index', '.geo', 'coordinates'])

# Optional: simulate labels
df['Fire_Label'] = ((df['NDVI'] < 0.2) & (df['NBR'] < 0.3)).astype(int)

print(df.head())
