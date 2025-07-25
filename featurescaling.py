import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the cleaned dataset
df = pd.read_csv("cleaned_small_dataset.csv")

# Select numerical columns for scaling
numerical_columns = ['duration', 'gross', 'num_voted_users', 'imdb_score']
scaler = MinMaxScaler()

# Scale the numerical features
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save the scaled dataset
df.to_csv("scaled_small_dataset.csv", index=False)
print("Feature scaling completed. Scaled dataset saved as 'scaled_small_dataset.csv'.")