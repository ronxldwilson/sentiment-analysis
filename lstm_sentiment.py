#!/usr/bin/env python3
"""
Deep Learning Sentiment Analysis using LSTM with TensorFlow/Keras
This script trains an LSTM model on movie reviews for sentiment classification.
"""

import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data
nltk.download('movie_reviews', quiet=True)

def load_data():
    """
    Load movie reviews data.
    Returns: list of (review_text, label) tuples where label is 0 (neg) or 1 (pos)
    """
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            text = movie_reviews.raw(fileid)
            label = 1 if category == 'pos' else 0
            documents.append((text, label))

    # Shuffle
    random.shuffle(documents)
    return documents

def preprocess_data(documents, max_words=10000, max_len=200):
    """
    Tokenize and pad sequences.
    """
    texts = [text for text, label in documents]
    labels = [label for text, label in documents]

    # Tokenize
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    return padded_sequences, np.array(labels), tokenizer

def build_model(max_words=10000, max_len=200):
    """
    Build LSTM model.
    """
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model():
    """
    Train the LSTM model.
    """
    print("Loading and preprocessing data...")
    documents = load_data()
    X, y, tokenizer = preprocess_data(documents)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Build model
    model = build_model()
    print("Model summary:")
    model.summary()

    # Train
    print("\nTraining model...")
    history = model.fit(X_train, y_train,
                       epochs=5,
                       batch_size=32,
                       validation_split=0.2,
                       verbose=1)

    # Evaluate
    print("\nEvaluating model...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # Analyze misclassifications
    misclassified_indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
    print(f"\nNumber of misclassifications: {len(misclassified_indices)} out of {len(y_test)}")

    # Show sample of misclassified reviews (need to decode sequences back to text)
    print("\nSample of misclassified reviews:")
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

    for i in misclassified_indices[:5]:  # Show first 5
        # Decode sequence back to words
        words = [reverse_word_index.get(word_id, '<OOV>') for word_id in X_test[i] if word_id != 0]
        review_text = ' '.join(words[:50])  # First 50 words

        print(f"\nTrue: {'POS' if y_test[i] == 1 else 'NEG'} | Predicted: {'POS' if y_pred[i] == 1 else 'NEG'}")
        print(f"Review: {review_text}...")

    return model, tokenizer

def classify_sentiment(text, model, tokenizer, max_len=200):
    """
    Classify sentiment of new text.
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    pred_prob = model.predict(padded, verbose=0)[0][0]
    pred = 1 if pred_prob > 0.5 else 0
    return 'POS' if pred == 1 else 'NEG', pred_prob

def main():
    print("Training LSTM Sentiment Analysis Model...")
    print("=" * 60)

    model, tokenizer = train_model()

    print("\n" + "=" * 60)
    print("Testing with sample reviews:")

    test_reviews = [
        "This movie was fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money.",
        "The acting was good but the plot was boring.",
        "Amazing storyline and great performances.",
        "I hated this movie. It was so bad."
    ]

    for review in test_reviews:
        sentiment, prob = classify_sentiment(review, model, tokenizer)
        print(f"\nReview: {review}")
        print(f"Predicted: {sentiment} (confidence: {prob:.3f})")

    print("\n" + "=" * 60)
    print("To classify your own reviews, use the classify_sentiment function.")

if __name__ == "__main__":
    main()
