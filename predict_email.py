import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

data = pd.read_csv("data/processed/email_clean.csv")

X = data["text_combined"]

vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(X)

model = joblib.load("models/email_model.pkl")

def predict_email(email):

    email_vector = vectorizer.transform([email])

    prediction = model.predict(email_vector)

    return prediction[0]