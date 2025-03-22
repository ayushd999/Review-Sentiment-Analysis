#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import pandas as pd
import requests
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Ensure model and vectorizer are loaded
def load_model():
    global model, vectorizer

    model_path = os.path.join(os.getcwd(), "model.pkl")
    vectorizer_path = os.path.join(os.getcwd(), "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("❌ Error: Model or Vectorizer file is missing! Ensure they are uploaded.")
        return

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    print("✅ Model and Vectorizer loaded successfully!")

load_model()

RECAPTCHA_SECRET_KEY = "YOUR_SECRET_KEY"  # Replace with your actual reCAPTCHA Secret Key

@app.route('/')
def index():
    if os.path.exists("stored_reviews.csv"):
        df_reviews = pd.read_csv("stored_reviews.csv")
    else:
        df_reviews = pd.DataFrame(columns=["Product", "Review", "Sentiment"])
    return render_template('index.html', reviews=df_reviews.to_dict(orient='records'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        review_text = request.form.get("review")
        product_name = request.form.get("product")
        recaptcha_response = request.form.get("g-recaptcha-response")  # Captcha response

        # Verify reCAPTCHA with Google
        recaptcha_verify_url = "https://www.google.com/recaptcha/api/siteverify"
        recaptcha_payload = {"secret": RECAPTCHA_SECRET_KEY, "response": recaptcha_response}
        recaptcha_result = requests.post(recaptcha_verify_url, data=recaptcha_payload).json()

        if not recaptcha_result.get("success"):
            return "Error: reCAPTCHA verification failed. Please try again.", 400

        if not review_text or not product_name:
            return "Error: Missing product name or review!", 400

        # Ensure vectorizer is loaded before using it
        if 'vectorizer' not in globals():
            return "Error: Vectorizer not loaded. Please check the model files.", 500

        vec = vectorizer.transform([review_text])  # Fix: Ensure vectorizer is loaded
        pred = model.predict(vec)[0]
        sentiment = "Positive" if pred == 1 else "Negative" if pred == 0 else "Neutral"

        # Save the review
        stored_reviews_path = "stored_reviews.csv"
        if not os.path.exists(stored_reviews_path):
            df_reviews = pd.DataFrame(columns=["Product", "Review", "Sentiment"])
        else:
            df_reviews = pd.read_csv(stored_reviews_path)

        new_entry = pd.DataFrame({"Product": [product_name], "Review": [review_text], "Sentiment": [sentiment]})
        df_reviews = pd.concat([df_reviews, new_entry], ignore_index=True)
        df_reviews.to_csv(stored_reviews_path, index=False)

        return render_template('index.html', prediction=sentiment, reviews=df_reviews.to_dict(orient='records'))

    except Exception as e:
        return f"Internal Server Error: {str(e)}", 500

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    stored_reviews_path = "stored_reviews.csv"

    if not os.path.exists(stored_reviews_path):
        return render_template('index.html', reviews=[])

    df_reviews = pd.read_csv(stored_reviews_path)
    results = df_reviews[df_reviews["Product"].str.contains(query, case=False, na=False)]
    return render_template('index.html', reviews=results.to_dict(orient='records'))

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))



