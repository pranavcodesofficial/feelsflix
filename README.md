# 🎬 FeelsFlix — Emotion-Based AI Movie Recommender System

**FeelsFlix** is an intelligent movie recommendation engine that understands your **emotions** and suggests the perfect film for your current mood. Unlike traditional systems that rely on genres or history, FeelsFlix dives deeper using **emotion detection** from text and ML models trained on curated emotional datasets.

---

## 💡 Overview

This project bridges **Natural Language Processing (NLP)** and **Recommender Systems** by aligning user inputs with the emotional tone of movies. It aims to build a deeply personalized, context-aware recommendation platform where **emotions drive discovery**.

---

## 🔥 Features

- 🎯 Emotion-driven movie recommendation engine  
- 📊 Cleaned and vectorized emotion + movie datasets  
- 🧠 ML models for mapping mood to movies  
- 🧹 Robust data preprocessing and feature scaling  
- 💬 Accepts text input like: “I’m anxious” or “Feeling nostalgic”  
- 🌐 Deployable with **Flask** or **Streamlit**  
- 📎 Grad-CAM support (optional for visual explainability)  
- ⚡ Future-ready: chatbot, voice, or real-time emotion capture

---

## 📁 Folder Structure

```bash
feelsflix/
├── data/
│   ├── cleaned_emotion_data.csv
│   ├── combined_emotion.csv
│   ├── moviedata.csv
│   └── [other smaller CSVs]
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
├── .gitignore
├── README.md
└── requirements.txt

🧠 ML & NLP Pipeline
• Text Cleaning: Lowercasing, punctuation removal, stopwords removal
• Vectorization: TF-IDF or Count Vectorizer
• Emotion Classification: Logistic Regression / Random Forest
• Recommendation Logic: Cosine Similarity or KNN
• Scaling: StandardScaler or MinMaxScaler

🎯 Future Roadmap
• 🤖 Real-time emotion prediction (text / webcam)
• 💬 LLM-based chatbot interface (GPT/Claude APIs)
• ☁️ Cloud deployment (Render / Vercel / HuggingFace Spaces)
• 🧠 Recommendation explainability (Grad-CAM, attention maps)
• 🧍 User profile learning for personalized ranking

👨‍💻 Author

Pranav A
AI & Data Science | LLM + ML Enthusiast
📧 pranavcodesofficial@gmail.com
🔗 https://www.linkedin.com/in/pranavcodesofficial/
