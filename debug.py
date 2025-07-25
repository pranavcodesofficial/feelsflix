import pandas as pd

# Load dataset
movie_df = pd.read_csv("small_moviedata.csv")

# Display column names
print("Column Names in Movie Dataset:")
print(movie_df.columns)