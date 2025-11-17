#!/usr/bin/env python3
"""
Simple Sentiment Analysis Script using TextBlob
"""

from textblob import TextBlob

def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text.

    Returns:
        tuple: (polarity, subjectivity, sentiment_label)
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Determine sentiment label
    if polarity > 0.1:
        sentiment_label = "Positive"
    elif polarity < -0.1:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return polarity, subjectivity, sentiment_label

def main():
    # Sample texts for analysis
    texts = [
        "I love this product!",
        "This is terrible.",
        "The weather is okay.",
        "I'm extremely happy with the service.",
        "This movie was awful.",
        "It's an average day.",
        "This is the worst fing product ever.",
    ]

    print("Sentiment Analysis Results:")
    print("=" * 40)

    for text in texts:
        polarity, subjectivity, label = analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Polarity: {polarity:.2f} ({label})")
        print(f"Subjectivity: {subjectivity:.2f}")
        print()

if __name__ == "__main__":
    main()
