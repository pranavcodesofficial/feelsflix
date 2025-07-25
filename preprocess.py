import pandas as pd

# Load the preprocessed dataset
data = pd.read_csv("preprocessed_emotion_data.csv")

# Check the first few rows to confirm the data looks correct
print(data.head())

# Display the number of rows and columns
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")