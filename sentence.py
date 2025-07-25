import pandas as pd

# Load the dataset
file_name = "preprocessed_emotion_data.csv"
emotion_data = pd.read_csv(file_name)

# Drop the 'sentence' column
emotion_data = emotion_data.drop(columns=["sentence"])

# Save the updated dataset (optional)
updated_file_name = "cleaned_emotion_data.csv"
emotion_data.to_csv(updated_file_name, index=False)

print(f"'sentence' column removed. Updated dataset saved as {updated_file_name}.")