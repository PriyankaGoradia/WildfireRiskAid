import pandas as pd
import ast
from sklearn.model_selection import train_test_split

# Load the CSV exported from GEE
df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Alberta_NDVI_NBR_Extracted_10k.csv")

# Convert geometry column to lat/lon
df['coordinates'] = df['.geo'].apply(lambda x: ast.literal_eval(x)['coordinates'])
df[['lon', 'lat']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)

# Drop unnecessary columns
df = df.drop(columns=['system:index', '.geo', 'coordinates'])

# Simulate Fire_Label
df['Fire_Label'] = ((df['NDVI'] < 0.2) & (df['NBR'] < 0.3)).astype(int)

# Split into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save to CSV
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Split complete: train.csv and test.csv saved.")
