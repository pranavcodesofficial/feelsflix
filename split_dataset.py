import pandas as pd
from sklearn.model_selection import train_test_split

# Load the scaled dataset
df = pd.read_csv("scaled_small_dataset.csv")

# Split into train (80%) and test (20%) sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the split datasets
train_df.to_csv("train_small_moviedata.csv", index=False)
test_df.to_csv("test_small_moviedata.csv", index=False)

print(f"Training set: {len(train_df)} rows")
print(f"Testing set: {len(test_df)} rows")