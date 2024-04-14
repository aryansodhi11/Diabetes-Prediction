# diabetes_model.py

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# Load the dataset
data = pd.read_csv(r"C:/Users/Bunny/Downloads/aryandiabetes project/diabetes prediction/diabetes prediction/diabetes.csv")

# Preprocess the data
features = data.drop("Outcome", axis=1)
labels = data["Outcome"]

# Feature scaling
scaler = StandardScaler()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Fit the scaler on the training data only
scaler.fit(X_train)

# Transform both the training and test sets
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train a K-Nearest Neighbors Classifier
start_time = time.time()  # Record the starting time
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X_train, y_train)

# Model evaluation
y_pred = knn.predict(X_test)
end_time = time.time()  # Record the ending time
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nTraining time:", end_time - start_time, "seconds")

# Save the model
dump(knn, "diabetes_model_knn.joblib")
