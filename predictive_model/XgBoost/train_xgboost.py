import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load training and testing data
train_df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Data-2023/train_complete_features_2023_data.csv")
test_df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/Data-2023/test_complete_features_2023_data.csv")


# Feature list (based on your actual columns)
features = [
    'NDVI', 'NBR', 'NDWI', 'Temp', 'Wind_Dir',
    'Wind_Spd', 'Humidity', 'Elev', 'Slope'
]

# Split features and labels
X_train = train_df[features]
y_train = train_df['Fire_Label']
X_test = test_df[features]
y_test = test_df['Fire_Label']

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Save model to file
joblib.dump(model, "xgboost_wildfire_model.joblib")
print("✅ Model saved as xgboost_wildfire_model.joblib")

# Predict and evaluate
y_pred = model.predict(X_test)
print("✅ XGBoost Evaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
