import pandas as pd

# Load datasets
movie_df = pd.read_csv("small_moviedata.csv")
emotion_df = pd.read_csv("small_emotiondata.csv")

# Step 1: Handle missing values
# Fill NaNs with appropriate values
movie_df.fillna(0, inplace=True)

# Step 2: Select essential features
columns_to_keep = [
    'director_name', 'duration', 'actor_1_name', 'imdb_score',
    'plot_keywords', 'country_USA', 'gross', 'num_voted_users'
]
movie_df = movie_df[columns_to_keep]

# Step 3: Merge datasets (assuming a common index)
merged_df = pd.concat([movie_df, emotion_df], axis=1)

# Step 4: Save the cleaned dataset
merged_df.to_csv("cleaned_small_dataset.csv", index=False)

print("Preprocessing completed. Cleaned dataset saved as 'cleaned_small_dataset.csv'.")
