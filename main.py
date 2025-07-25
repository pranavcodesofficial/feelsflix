from tmdbv3api import TMDb, Movie
import streamlit as st
import pandas as pd
import pickle
import joblib

# Load models and data
vectorizer = joblib.load("tfidf_vectorizer.pkl")

with open("emotion_model.pkl", "rb") as f:
    emotion_model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

movie_df = pd.read_csv("scaled_small_moviedata.csv")

# TMDb setup
tmdb = TMDb()
tmdb.api_key = "fb795794fc3f2326d481e15a59d1afe8"
movie_api = Movie()

def get_movie_info(title):
    try:
        result = movie_api.search(title)
        if result:
            poster_path = result[0].poster_path
            vote = result[0].vote_average
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/200x300?text=No+Poster"
            return poster_url, vote
        else:
            return "https://via.placeholder.com/200x300?text=No+Poster", "N/A"
    except:
        return "https://via.placeholder.com/200x300?text=No+Poster", "N/A"

# Emotion to genre mapping
emotion_to_genre = {
    "joy": "Comedy",
    "anger": "Action",
    "sadness": "Drama",
    "fear": "Thriller",
    "surprise": "Mystery",
    "love": "Romance",
    "neutral": "Adventure"
}

def predict_emotion(text):
    vect_text = vectorizer.transform([text])
    pred = emotion_model.predict(vect_text)
    emotion = label_encoder.inverse_transform(pred)[0]
    return emotion

def recommend_movies(emotion, top_n=5):
    emotion_keywords = {
        "joy": ['comedy', 'fun', 'joy', 'laugh', 'happy', 'family', 'friendship'],
        "anger": ['action', 'fight', 'revenge', 'battle', 'war', 'crime'],
        "sadness": ['drama', 'loss', 'grief', 'family', 'emotional', 'tragic', 'life'],
        "fear": ['thriller', 'horror', 'escape', 'dark', 'danger', 'mystery'],
        "surprise": ['mystery', 'unexpected', 'twist', 'thriller', 'puzzle'],
        "love": ['romance', 'love', 'relationship', 'heart', 'affair', 'couple'],
        "neutral": ['adventure', 'journey', 'explore', 'quest', 'travel', 'epic']
    }

    keywords = emotion_keywords.get(emotion, ['drama'])
    pattern = '|'.join(keywords)
    recommended = movie_df[movie_df['plot_keywords'].str.contains(pattern, na=False, case=False)]

    recommended = recommended.sample(frac=1).reset_index(drop=True)
    top_movies = recommended.sort_values(by="imdb_score", ascending=False).head(top_n)
    return top_movies[['movie_title', 'plot_keywords', 'imdb_score']]

# Streamlit app UI
st.title("🎬 Mood-Based Movie Recommender")
st.write("Describe how you're feeling and get movie suggestions that match your mood!")

user_input = st.text_input("How are you feeling today?")

if st.button("Recommend"):
    if user_input.strip() == "":
        st.warning("Please enter something about your mood.")
    else:
        emotion = predict_emotion(user_input)
        st.success(f"Detected Emotion: **{emotion}**")
        st.subheader("🎥 Recommended Movies:")

        results = recommend_movies(emotion)
        for idx, row in results.iterrows():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                poster_url, actual_rating = get_movie_info(row["movie_title"])
                st.image(poster_url, width=150)
            
            with col2:
                st.markdown(f"### 🎬 {row['movie_title']}")
                st.markdown(f"📝 *{row['plot_keywords'].replace('|', ', ')}*")
                st.markdown(f"⭐ **IMDb Score:** {actual_rating}/10")
            
            st.markdown("---")