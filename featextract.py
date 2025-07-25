import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the preprocessed dataset
df = pd.read_csv("cleaned_emotion_data.csv")
df['cleaned_text'] = df['cleaned_text'].fillna('')  # Replace NaN with empty strings

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # Convert words into numerical features
X = vectorizer.fit_transform(df['cleaned_text'])

# Save the vectorizer for later use
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Save processed features as a DataFrame
X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Combine with labels
X_df['emotion'] = df['emotion']

# Save the final dataset
X_df.to_csv("vectorized_emotion_data.csv", index=False)

print("Feature extraction completed and saved!")