import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load the dataset
file_name = "moviedata.csv"
movies_df = pd.read_csv(file_name)


# Step 3: Handle Missing Values
# Fill missing values in numerical columns with the median
numerical_cols = ["duration", "gross", "num_voted_users", "title_year", "imdb_score"]
movies_df[numerical_cols] = movies_df[numerical_cols].fillna(movies_df[numerical_cols].median())

# Fill missing values in one-hot encoded genre columns with 0
genre_columns = [col for col in movies_df.columns if col.startswith("genre_")]
movies_df[genre_columns] = movies_df[genre_columns].fillna(0)

# Step 4: Normalize Numerical Features
scaler = MinMaxScaler()
movies_df[["duration", "gross", "num_voted_users", "title_year", "imdb_score"]] = scaler.fit_transform(
    movies_df[["duration", "gross", "num_voted_users", "title_year", "imdb_score"]]
)

# Step 5: Split the Data into Features and Target
# Target: IMDb score (can be replaced with other targets)
X = movies_df.drop(columns=["imdb_score"])  # Features
y = movies_df["imdb_score"]  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Step 8: Save the Preprocessed Dataset (Optional)
preprocessed_file_name = "preprocessed_moviedata.csv"
movies_df.to_csv(preprocessed_file_name, index=False)
print(f"Preprocessed data saved as '{preprocessed_file_name}'")