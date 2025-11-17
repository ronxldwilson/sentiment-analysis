#!/usr/bin/env python3
"""
Machine Learning-based Sentiment Analysis using NLTK Movie Reviews and Scikit-learn
This script trains a Naive Bayes classifier on movie reviews to predict positive/negative sentiment.
"""

import joblib
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data
nltk.download('movie_reviews', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove punctuation, tokenize, remove stopwords, lemmatize.
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def load_data():
    """
    Load and preprocess movie reviews data.
    Returns: list of (review_text, label) tuples
    """
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            text = movie_reviews.raw(fileid)
            label = category  # 'pos' or 'neg'
            documents.append((text, label))

    # Shuffle the documents
    random.shuffle(documents)
    return documents

def train_model():
    """
    Train the sentiment analysis model.
    Returns: vectorizer, classifier
    """
    print("Loading and preprocessing data...")
    documents = load_data()

    # Preprocess texts
    texts = [preprocess_text(text) for text, label in documents]
    labels = [label for text, label in documents]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Vectorize with n-grams for better features
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))  # unigrams and bigrams
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier - try SVM for potentially better accuracy
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(random_state=42)
    classifier.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # Analyze misclassifications
    misclassified_indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
    print(f"\nNumber of misclassifications: {len(misclassified_indices)} out of {len(y_test)}")

    # Show sample of misclassified reviews
    print("\nSample of misclassified reviews:")
    for i in misclassified_indices[:10]:  # Show first 10
        print(f"\nTrue: {y_test[i].upper()} | Predicted: {y_pred[i].upper()}")
        print(f"Review: {X_test[i][:100]}...")  # Show first 200 chars

    #save model

    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(classifier, 'classifier.joblib')

    return vectorizer, classifier

def classify_sentiment(text, vectorizer, classifier):
    """
    Classify sentiment of new text.
    """
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    prediction = classifier.predict(vectorized)[0]
    return prediction

def main():
    print("Training Sentiment Analysis Model...")
    print("=" * 50)

    vectorizer, classifier = train_model()

    print("\n" + "=" * 50)
    print("Testing with sample reviews:")

    test_reviews = [
        "This movie was fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money.",
        "The acting was good but the plot was boring.",
        "Amazing storyline and great performances.",
        "I hated this movie. It was so bad."
    ]

    for review in test_reviews:
        sentiment = classify_sentiment(review, vectorizer, classifier)
        print(f"\nReview: {review}")
        print(f"Predicted Sentiment: {sentiment.upper()}")

    print("\n" + "=" * 50)
    print("To classify your own reviews, modify the test_reviews list in the script.")

if __name__ == "__main__":
    main()
