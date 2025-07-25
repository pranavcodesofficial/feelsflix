import pandas as pd

# Read original datasets
movie_df = pd.read_csv("moviedata.csv")
emotion_df = pd.read_csv("cleaned_emotion_data.csv")

# Limit datasets to 4,000 rows each
small_movie_df = movie_df.sample(n=4000, random_state=42) if len(movie_df) > 4000 else movie_df
small_emotion_df = emotion_df.sample(n=4000, random_state=42) if len(emotion_df) > 4000 else emotion_df

# Save the smaller datasets
small_movie_df.to_csv("small_moviedata.csv", index=False)
small_emotion_df.to_csv("small_emotiondata.csv", index=False)

print(f"Reduced movie dataset to {len(small_movie_df)} rows.")
print(f"Reduced emotion dataset to {len(small_emotion_df)} rows.")