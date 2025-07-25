import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the cleaned dataset
df = pd.read_csv("small_moviedata.csv")

# Features to scale
numeric_features = ["duration", "imdb_score", "num_voted_users"]
scaler = MinMaxScaler()

# Scale only if columns exist
for feature in numeric_features:
    if feature in df.columns:
        df[feature] = scaler.fit_transform(df[[feature]])

# Save scaled dataset
df.to_csv("scaled_small_moviedata.csv", index=False)

print(f"Dataset scaled and saved as 'scaled_small_moviedata.csv'.")