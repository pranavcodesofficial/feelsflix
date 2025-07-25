import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import scipy.sparse

# Load data
X = scipy.sparse.load_npz("X_tfidf.npz")
y = pd.read_csv("y_labels.csv").values.ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save the trained model
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)