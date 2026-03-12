import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib

# Load the data
df = pd.read_csv('heart.csv')

# Prepare features and target
scal = MinMaxScaler()
feat = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
df[feat] = scal.fit_transform(df[feat])

y = df["target"]
X = df.drop('target', axis=1)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train model
Knn_clf = KNeighborsClassifier(n_neighbors=7)
Knn_clf.fit(X_train, Y_train)

# Test model
y_pred = Knn_clf.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Save model
joblib.dump(Knn_clf, 'heartmodel.pkl')
print("Model saved successfully as heartmodel.pkl")
