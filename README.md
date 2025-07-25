# 🎬 FeelsFlix — Emotion-Based AI Movie Recommender System

FeelsFlix is an intelligent movie recommendation engine that understands your **emotions** and suggests the perfect film for your current mood. Unlike traditional systems that rely on genre or past watch history, FeelsFlix dives deeper by using **emotion detection** from text and **machine learning models** trained on curated datasets.

---

## 💡 Overview

This project bridges **Natural Language Processing (NLP)** and **Movie Recommendation** by aligning user emotions with the emotional tone of movies. It aims to build a deeply **personalized**, context-aware recommendation system where emotions drive discovery.

---

## 🔥 Features

- 🎯 **Emotion-based recommendation engine**
- 📊 Cleaned and vectorized emotion & movie datasets
- 🧠 Machine Learning models for mood-to-movie mapping
- 🧹 Data preprocessing and feature scaling
- 🌐 Easily deployable with Flask or Streamlit
- 💬 Text input like “I’m anxious” or “Feeling nostalgic” accepted
- 📎 Grad-CAM support (optional for visualization on medical/vision extensions)
- ⚡ Scalable to real-time chat or voice inputs in future versions

---

## 📁 Folder Structure
feelsflix/
├── data/
│   ├── cleaned_emotion_data.csv
│   ├── combined_emotion.csv
│   ├── moviedata.csv
│   └── other smaller CSVs
├── models/
│   ├── emotion_classifier.pkl
│   └── movie_recommender.pkl
├── notebooks/
│   ├── 1_data_cleaning.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_model_training.ipynb
│   └── 4_recommendation_testing.ipynb
├── src/
│   ├── recommender.py
│   ├── emotion_utils.py
│   └── model_utils.py
├── app/
│   └── app.py  # Streamlit or Flask frontend
├── README.md
├── requirements.txt
└── .gitignore

 🧠 ML & NLP Pipeline
	•	Text Cleaning: Lowercasing, punctuation removal, stopwords
	•	Vectorization: TF-IDF or Count Vectorizer
	•	Emotion Classification: Logistic Regression / RandomForest
	•	Movie Recommendation: Cosine Similarity / KNN
	•	Scalers: StandardScaler / MinMaxScaler for numeric features
 
 🎯 Future Roadmap
	•	Integrate real-time emotion prediction via webcam/text
	•	Add chatbot support (LLM-based input)
	•	Cloud deployment (e.g., Render, Vercel, HuggingFace Spaces)
	•	Recommendation explainability with attention/Grad-CAM (NLP)
	•	User preference learning for dynamic ranking
 
 👨‍💻 Author
 Pranav A
 AI + Data Science | LLM + ML Enthusiast
 📧 pranavcodesofficial@gmail.com
 https://www.linkedin.com/in/pranavcodesofficial/

