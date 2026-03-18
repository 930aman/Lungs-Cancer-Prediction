import joblib
import pandas as pd
import os
import sys

# Define paths
base_path = r"C:\PYTHON PROGRAMMING\Lungs Cancer Prediction"
model_path = os.path.join(base_path, "lung_cancer_rf_model.joblib")
scaler_path = os.path.join(base_path, "lung_cancer_scaler.joblib")

def verify():
    print(f"Checking for model at: {model_path}")
    if not os.path.exists(model_path):
        print("Model file missing.")
        return False
        
    print(f"Checking for scaler at: {scaler_path}")
    if not os.path.exists(scaler_path):
        print("Scaler file missing.")
        # Attempt to run without scaler if model exists? 
        # Usually random forest doesn't strictly need scaling, but if it was pipelined...
    
    try:
        print("Loading model...")
        model = joblib.load(model_path)
        print("Model loaded.")
        
        scaler = None
        if os.path.exists(scaler_path):
            print("Loading scaler...")
            scaler = joblib.load(scaler_path)
            print("Scaler loaded.")
            
        # Sample data derived from data.csv (Row 0)
        # Index and Patient Id are excluded
        sample_data = {
            'Age': [33], 
            'Gender': [1], 
            'Air Pollution': [2], 
            'Alcohol use': [4],
            'Dust Allergy': [5], 
            'OccuPational Hazards': [4], 
            'Genetic Risk': [3],
            'chronic Lung Disease': [2], 
            'Balanced Diet': [2], 
            'Obesity': [4],
            'Smoking': [3], 
            'Passive Smoker': [2], 
            'Chest Pain': [2],
            'Coughing of Blood': [4], 
            'Fatigue': [3], 
            'Weight Loss': [4],
            'Shortness of Breath': [2], 
            'Wheezing': [2], 
            'Swallowing Difficulty': [3],
            'Clubbing of Finger Nails': [1], 
            'Frequent Cold': [2], 
            'Dry Cough': [3], 
            'Snoring': [4]
        }
        
        df = pd.DataFrame(sample_data)
        print("Sample data frame created.")
        
        input_data = df
        if scaler:
            try:
                print("Attempting to scale data...")
                # Check if scaler expects the same number of features
                # There might be a mismatch if scaler was trained on more/fewer columns
                input_data = scaler.transform(df)
                print("Data scaled.")
            except Exception as e:
                print(f"Scaling failed: {e}")
                print("Trying predicting without scaling...")
                input_data = df

        print("Attempting prediction...")
        prediction = model.predict(input_data)
        print(f"Prediction verification successful! Result: {prediction[0]}")
        return True

    except Exception as e:
        print(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    if verify():
        sys.exit(0)
    else:
        sys.exit(1)
