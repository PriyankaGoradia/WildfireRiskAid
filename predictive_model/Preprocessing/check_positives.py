import pandas as pd

# Load your labeled dataset
df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Data-2023/preprocessed_complete_features_with_real_fire_labels.csv")

# Ensure Timestamp is parsed as datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Define training and testing date ranges
train_range = ('2023-05-01', '2023-07-31')
test_range = ('2023-08-01', '2023-09-30')

# Filter based on date ranges
train_df = df[(df['Timestamp'] >= train_range[0]) & (df['Timestamp'] <= train_range[1])]
test_df = df[(df['Timestamp'] >= test_range[0]) & (df['Timestamp'] <= test_range[1])]

# Count positives
train_positives = train_df['Fire_Label'].sum()
test_positives = test_df['Fire_Label'].sum()

# Total counts
print(f"🔍 Training Data ({train_range[0]} to {train_range[1]}):")
print(f"   ➤ Total samples: {len(train_df)}")
print(f"   ➤ Positive fire labels: {train_positives}")

print(f"\n🧪 Testing Data ({test_range[0]} to {test_range[1]}):")
print(f"   ➤ Total samples: {len(test_df)}")
print(f"   ➤ Positive fire labels: {test_positives}")
