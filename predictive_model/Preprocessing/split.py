import pandas as pd
from sklearn.model_selection import train_test_split

# Load the preprocessed labeled dataset
df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Data-2023/preprocessed_complete_features_with_real_fire_labels.csv")

# Stratified split to maintain fire/non-fire distribution
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['Fire_Label'],
    random_state=42
)

# Save the splits
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("✅ Dataset split complete:")
print(f"Training set: {len(train_df)} rows")
print(f"Testing set: {len(test_df)} rows")
