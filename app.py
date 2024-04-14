# app.py
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load
import numpy as np
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app,origins='*')  # Enable CORS for all routes

# Load the model
clf = load("diabetes_model_knn.joblib")

# Load the scaler
scaler = StandardScaler()

data = pd.read_csv(r"C:/Users/tanis/Documents/BrainoVision/aryandiabetes project/diabetes prediction/diabetes prediction/diabetes.csv")

# Preprocess the data
features = data.drop("Outcome", axis=1)
labels = data["Outcome"]

# Feature scaling
scaler = StandardScaler()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Fit the scaler on the training data only
scaler.fit(X_train)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Ensure all features are numeric and handle any non-numeric values
    features = np.array([
        float(data["Pregnancies"]),
        float(data["Glucose"]),
        float(data["BloodPressure"]),
        float(data["SkinThickness"]),
        float(data["Insulin"]),
        float(data["BMI"]),
        float(data["DiabetesPedigreeFunction"]),
        float(data["Age"]),
    ]).reshape(1, -1)  # Reshape to match expected input shape

    # Apply the same scaler used during training
    features_scaled = scaler.transform(features)

    # Predict
    prediction = clf.predict(features_scaled)

    # Format and return the response
    return jsonify({"prediction": int(prediction[0])})  # Convert prediction to int

if __name__ == "__main__":
    app.run(debug=True)
