# app.py
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd

# --- NLTK setup (download once, quiet) ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- Load model and vectorizer (only once) ---
lr_model = pickle.load(open('logistic_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# --- Prepare stopwords and lemmatizer ---
stop_words = set(stopwords.words('english'))
stop_words.add('im')  # optional: add more custom stopwords
lemmatizer = WordNetLemmatizer()

# --- Text cleaning & lemmatization function ---
def clean_input(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # remove links
    text = re.sub(r'@\w+', '', text)           # remove mentions
    text = re.sub(r'#\w+', '', text)           # remove hashtags
    text = re.sub(r'[^a-z\s]', ' ', text)      # remove punctuation/numbers
    words = [w for w in text.split() if w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized)

# --- Prediction function ---
def predict_sentiment(text):
    clean_text = clean_input(text)
    vectorized = vectorizer.transform([clean_text])
    prediction = lr_model.predict(vectorized)
    return prediction[0]

# --- Streamlit UI ---
st.title("📊 Tweet Sentiment Analyzer")

# Input from user
user_input = st.text_area("Enter your tweet here:")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # --- Predict sentiment ---
        result_numeric = predict_sentiment(user_input)
        # Map numeric labels to text
        label_map = { -1: "Negative", 0: "Neutral", 1: "Positive" }
        result_text = label_map[result_numeric]

        # --- Display sentiment with color ---
        if result_numeric == 1:
            st.success(f"😃 Predicted Sentiment: {result_text}")
        elif result_numeric == -1:
            st.error(f"😔 Predicted Sentiment: {result_text}")
        else:
            st.info(f"😐 Predicted Sentiment: {result_text}")