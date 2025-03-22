#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os


# In[ ]:




app = Flask(__name__)


def load_model():
    global model, vectorizer
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        main()
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


def main():
    # Load dataset
    csv_path = r"customer_reviews_sentiment_words.csv"
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="utf-8")

    # Check if required columns exist
    if "review_text" not in df.columns or "sentiment" not in df.columns:
        print("Error: CSV file must contain 'review_text' and 'sentiment' columns.")
        sys.exit(1)

    x = df["review_text"]
    y = df["sentiment"]

    # Vectorize text data
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    x_vector = vectorizer.fit_transform(x)

    # Train model
    xtrain, xtest, ytrain, ytest = train_test_split(x_vector, y, train_size=0.8, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(xtrain, ytrain)

    # Save model and vectorizer as pickle files
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    # Create storage for reviews
    stored_reviews_path = "stored_reviews.csv"
    if not os.path.exists(stored_reviews_path):
        pd.DataFrame(columns=["Product", "Review", "Sentiment"]).to_csv(stored_reviews_path, index=False)

    print("Pickle files 'model.pkl' and 'vectorizer.pkl' created successfully!")
    print("Storage file 'stored_reviews.csv' created successfully!")


@app.route('/')
def index():
    df_reviews = pd.read_csv("stored_reviews.csv")
    return render_template('index.html', reviews=df_reviews.to_dict(orient='records'))


@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form['review']
    product_name = request.form['product']
    vec = vectorizer.transform([review_text])
    pred = model.predict(vec)[0]
    sentiment = "Positive" if pred == 1 else "Negative" if pred == 0 else "Neutral"

    # Store the review
    df_reviews = pd.read_csv("stored_reviews.csv")
    new_entry = pd.DataFrame({"Product": [product_name], "Review": [review_text], "Sentiment": [sentiment]})
    df_reviews = pd.concat([df_reviews, new_entry], ignore_index=True)
    df_reviews.to_csv("stored_reviews.csv", index=False)

    return render_template('index.html', prediction=sentiment, reviews=df_reviews.to_dict(orient='records'))


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    df_reviews = pd.read_csv("stored_reviews.csv")
    results = df_reviews[df_reviews["Product"].str.contains(query, case=False, na=False)]
    return render_template('index.html', reviews=results.to_dict(orient='records'))


if __name__ == "__main__":
    load_model()
    app.run()


