# 🐦 Tweet Sentiment Analyzer

A **Streamlit web application** to predict the **sentiment of tweets**. This project uses a **Logistic Regression model** with **TF-IDF features** to classify tweets into **Positive, Neutral, or Negative** categories. It also displays the **probabilities for each class** in an interactive bar chart.<br> 
**Data link** : https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

---

## 🔹 Features

- Preprocess tweets: remove stopwords, URLs, mentions, hashtags, and punctuation  
- Lemmatization for better text normalization   
- TF-IDF vectorizer for feature extraction
- Logistic Regression model trained on Twitter dataset 
- Color-coded sentiment output in Streamlit  
- Easy-to-use web interface  

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application |
| `logistic_model.pkl` | Trained Logistic Regression model |
| `tfidf_vectorizer.pkl` | Pickled TF-IDF vectorizer |
| `requirement.txt` | Python dependencies |
| `twitter_training.csv` | Training dataset |
| `sentiment_analysis.ipynb` | Notebook with data preprocessing and model training |
| `README.md` | Project description and instructions |

---

## 💻 Installation

1. **Clone the repository**

```bash
git clone https://github.com/MohamedAli1937/Sentiment-Analysis-Web-App.git
```
## ⚙️ Install dependencies
```bash
pip install -r requirements.txt
```
## 🎮 Run the Streamlit app
```bash
streamlit run app.py
```
## 🤔 Prediction Function
```python
def predict_sentiment(text):
    clean_text = clean_tweet_stopword_lemmatize(text)  # your cleaning + lemmatization function
    vectorized = vectorizer.transform([clean_text])
    return lr_model.predict(vectorized)[0]
```
---
## 🧠 How It Works


1.  **Preprocessing**:
- Lowercasing, removing URLs, mentions, hashtags, punctuation
- Stopwords removal and lemmatization   
2.  **Feature Extraction**:
- TF-IDF converts text to numerical vectors
3.  **Model**:
  - Logistic Regression predicts sentiment class
4.  **Output**:
- Sentiment class (Positive/Neutral/Negative)
---
## 🚀 Future Improvements
1.  **Better Models**
- Use **DistilBERT** or **RoBERTa** for more accurate predictions
- Deep learning models capture context better than Logistic Regression
2.  **Emotion Detection**
- Expand beyond Positive/Neutral/Negative
- Detect specific emotions: Happy, Sad, Angry, Fear, Surprise, etc.
- Use libraries like NRCLex or train multi-class classifiers
3.  **Data Enhancements**
- Add more **neutral tweets** to improve model balance
- Include tweets in multiple languages
4.  **UI/UX Improvements**
- Show **word clouds** for positive/negative words
- Display **historical sentiment trends** from multiple tweets
- Add **interactive charts** for probabilities
5.  **Deployment**
- Deploy online via **Streamlit Cloud**, **Heroku**, or **AWS**
- Make a **public demo** for users to try
---
## ⚠️ Known Limitations

- The current Logistic Regression model sometimes misclassifies neutral tweets as positive or negative.

- This happens because the training dataset has fewer neutral examples, making the model biased toward positive/negative sentiment.

- Probabilities for neutral predictions may be less reliable compared to positive or negative.
