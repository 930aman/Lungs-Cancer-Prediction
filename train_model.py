import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define paths
base_path = r"C:\PYTHON PROGRAMMING\Lungs Cancer Prediction"
data_path = os.path.join(base_path, "data.csv")
model_path = os.path.join(base_path, "lung_cancer_rf_model.joblib")
scaler_path = os.path.join(base_path, "lung_cancer_scaler.joblib")
encoder_path = os.path.join(base_path, "lung_cancer_encoder.joblib")

print("Loading data...")
df = pd.read_csv(data_path)

# Drop irrelevant columns
columns_to_drop = ['index', 'Patient Id']
df = df.drop(columns=columns_to_drop, axis=1)

# Separate Features and Target
X = df.drop('Level', axis=1)
y = df['Level']

# Encode Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Target classes: {le.classes_}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features (Optional for RF, but good practice if we want to swap models later)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save artifacts
print("Saving model and artifacts...")
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(le, encoder_path)
print("Done.")
