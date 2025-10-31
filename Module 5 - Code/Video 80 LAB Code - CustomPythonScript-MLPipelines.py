import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Read the dataset
dataset = pd.read_csv('dataset_sdn.csv')

# Separate features (X) and target variable (y)
X = dataset[['dt', 'switch', 'src', 'dst', 'pktcount', 'bytecount', 'dur', 'dur_nsec', 'tot_dur', 'flows', 'packetins', 'pktperflow', 'byteperflow', 'pktrate', 'Pairflow', 'Protocol', 'port_no', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps']].rename(columns=lambda x: x.strip())

y = dataset['label']

# Convert categorical variables to numerical using One-Hot Encoding
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Machine Learning Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test) 

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# Classification Report
print("\nClassification Report:")
print("Precision measure the accuracy of positive predicitons, Recall measures the fraction of True Positives Detected, and F1-Score is the Harmonic Mean of Precision and Recall")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print("The Confusion Matrix shows the number of True Negatives, False Positives, False Negatives, and True Positives.")
print(conf_matrix)

