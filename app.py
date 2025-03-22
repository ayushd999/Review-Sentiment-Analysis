#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import pickle
import sys
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Ensure model and vectorizer exist
def load_model():
    global model, vectorizer
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        train_and_save_model()  # If missing, train model

    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Train and save model
def train_and_save_model():
    csv_path = "customer_reviews_sentiment_words.csv"
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found!")
        return

    df = pd.read_csv(csv_path, encoding="utf-8")

    if "review_text" not in df.columns or "sentiment" not in df.columns:
        print("Error: CSV must contain 'review_text' and 'sentiment' columns.")
        return

    x = df["review_text"]
    y = df["sentiment"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    x_vector = vectorizer.fit_transform(x)

    xtrain, xtest, ytrain, ytest = train_test_split(x_vector, y, train_size=0.8, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(xtrain, ytrain)

    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("✅ Model and vectorizer saved!")

    # Ensure storage file exists
    stored_reviews_path = "stored_reviews.csv"
    if not os.path.exists(stored_reviews_path):
        pd.DataFrame(columns=["Product", "Review", "Sentiment"]).to_csv(stored_reviews_path, index=False)

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
        
        if not review_text or not product_name:
            return "Error: Missing product name or review!", 400

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



