import pandas as pd

# Load dataset
file_path = "small_moviedata.csv"
df = pd.read_csv(file_path)

# Keep only important columns
columns_to_keep = [
    "director_name", "duration", "actor_1_name", "genres", "imdb_score", "movie_title", "num_voted_users"
]
df = df[columns_to_keep]

# Convert object columns to category for efficiency
for col in ["director_name", "actor_1_name", "genres", "movie_title"]:
    df[col] = df[col].astype("category")

# Reduce float precision
df["imdb_score"] = df["imdb_score"].astype("float32")
df["duration"] = df["duration"].astype("float32")
df["num_voted_users"] = df["num_voted_users"].astype("int32")

# Save cleaned dataset
df.to_csv("optimized_small_moviedata.csv", index=False)

print(f"Optimized dataset saved with {len(df)} rows and {len(df.columns)} columns.")