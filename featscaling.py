import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset
df = pd.read_csv("moviedata.csv")

# Select numerical and categorical features
numerical_features = ["duration", "imdb_score", "num_voted_users", "gross"]
categorical_features = ["genres", "language", "country"]

# Standardize numerical features (Z-score scaling)
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# One-Hot Encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_cats = encoder.fit_transform(df[categorical_features])

# Convert encoded categories into a DataFrame
cat_columns = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_cats, columns=cat_columns)

# Concatenate original dataframe with encoded categories and drop old categorical columns
df = pd.concat([df, encoded_df], axis=1).drop(columns=categorical_features)

# Save the processed data
df.to_csv("moviedata.csv", index=False)

print("✅ Feature Scaling & Encoding Done! Data saved as 'processed_movies.csv'.")