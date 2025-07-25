import pandas as pd

# Load the movie dataset
movies_df = pd.read_csv("moviedata.csv")
print(movies_df.head())
# Replace the '|' separator with ',' (or split by '|' directly)
movies_df['genres'] = movies_df['genres'].str.replace('|', ',', regex=False)

# Now proceed with the one-hot encoding process
genres_split = movies_df['genres'].str.split(',', expand=True).stack().unique()

# Create a list of DataFrames with one-hot encoded columns for each genre
genre_columns = []
for genre in genres_split:
    genre_column = movies_df['genres'].apply(lambda x: 1 if genre in x else 0)
    genre_columns.append(genre_column.rename(f'genre_{genre.strip()}'))

# Concatenate all genre columns at once
movies_df = pd.concat([movies_df] + genre_columns, axis=1)

# Check the resulting DataFrame
print(movies_df.head())