# Lung Cancer Prediction AI App

This project contains a Machine Learning application that predicts the risk level of lung cancer based on patient data.

## Features
- **Interactive Web Interface**: Built with [Streamlit](https://streamlit.io/).
- **Real-time Prediction**: Uses a Random Forest Classifier to predict risk (Low, Medium, High).
- **Customizable Inputs**: Adjust patient age, lifestyle habits, and symptoms to see how they affect risk.

## Installation

1. Open your terminal/command prompt.
2. Navigate to the project directory:
   ```bash
   cd "C:\PYTHON PROGRAMMING\Lungs Cancer Prediction"
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Run the application using the following command:
   ```bash
   streamlit run app.py
   ```
2. A browser window will open automatically with the application.
3. Use the sidebar to input patient data and click **"Predict Risk Level"** to see the result.

## Files
- `app.py`: The main application script.
- `train_model.py`: Script used to train the model (run this if model artifacts are missing).
- `data.csv`: Source dataset.
- `*.joblib`: Serialized model artifacts (Model, Scaler, Encoder).
