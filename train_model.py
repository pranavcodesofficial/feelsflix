import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib

# Load training dataset
train_df = pd.read_csv("train_small_moviedata.csv")

# Select features for training
features = ["duration", "imdb_score", "num_voted_users"]
X_train = train_df[features]

# Train KNN model
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(X_train)

# Save trained model
joblib.dump(knn, "knn_movie_model.pkl")

print("🎬 KNN model trained and saved as 'knn_movie_model.pkl'.")