from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os

VECTOR_FILE = "vectorized_emotion_data.csv"

def extract_features(df):
    """Applies TF-IDF vectorization efficiently to avoid memory issues."""
    print("🔹 Extracting features using TF-IDF... (Optimized for memory)")

    # Limit vocabulary to reduce memory usage
    vectorizer = TfidfVectorizer(max_features=10000)  # Adjust feature size as needed

    # Convert text to lower case and ensure all values are strings
    df['cleaned_text'] = df['cleaned_text'].astype(str).str.lower()

    # Fit and transform data
    X = vectorizer.fit_transform(df['cleaned_text'])

    # Convert to DataFrame
    feature_names = vectorizer.get_feature_names_out()
    X_df = pd.DataFrame(X.toarray(), columns=feature_names)

    # Save in smaller chunks
    print(f"✅ Feature extraction completed. Saving in chunks to '{VECTOR_FILE}'...")
    X_df.to_csv(VECTOR_FILE, index=False, chunksize=10000)

    print(f"✅ Data saved successfully. Feature matrix shape: {X.shape}")