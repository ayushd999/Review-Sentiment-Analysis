#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import pickle
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Google reCAPTCHA Secret Key
RECAPTCHA_SECRET_KEY = "6LdW3vwqAAAAADj1Aq6ZItg_cXvmIjB9brlgc1-F"

# Load model & vectorizer
def load_model():
    global model, vectorizer

    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"

    if not os.path.exists(model_path):
        print("❌ Error: model.pkl is missing!")
        return
    
    if not os.path.exists(vectorizer_path):
        print("❌ Error: vectorizer.pkl is missing!")
        return

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    print("✅ Model and Vectorizer loaded successfully!")

load_model()

@app.route('/')
def index():
    df_reviews = pd.read_csv("stored_reviews.csv")
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
            return jsonify({"error": "reCAPTCHA verification failed. Please try again."}), 400

        if not review_text or not product_name:
            return jsonify({"error": "Missing product name or review!"}), 400

        # Ensure vectorizer is loaded before using it
        if 'vectorizer' not in globals():
            return jsonify({"error": "Vectorizer not loaded. Please check the model files."}), 500

        vec = vectorizer.transform([review_text])  
        pred = model.predict(vec)[0]
        sentiment = "Positive" if pred == 1 else "Negative" if pred == 0 else "Neutral"

        # Store the review
        df_reviews = pd.read_csv("stored_reviews.csv")
        new_entry = pd.DataFrame({"Product": [product_name], "Review": [review_text], "Sentiment": [sentiment]})
        df_reviews = pd.concat([df_reviews, new_entry], ignore_index=True)
        df_reviews.to_csv("stored_reviews.csv", index=False)

        return render_template('index.html', prediction=sentiment, reviews=df_reviews.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    df_reviews = pd.read_csv("stored_reviews.csv")
    results = df_reviews[df_reviews["Product"].str.contains(query, case=False, na=False)]
    return render_template('index.html', reviews=results.to_dict(orient='records'))

if __name__ == "__main__":
    load_model()
    app.run()
