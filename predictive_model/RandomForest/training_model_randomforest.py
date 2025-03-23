import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the preprocessed data
df = pd.read_csv("/workspaces/WildfireRiskAid/predictive_model/train.csv")

# Define features and labels
X = df[['NDVI', 'NBR']]  # Add NDWI or other features if available
y = df['Fire_Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model to disk
joblib.dump(model, "fire_risk_model.pkl")
print("\nModel saved as fire_risk_random_forest_model.pkl")
