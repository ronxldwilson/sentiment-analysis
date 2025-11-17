#!/usr/bin/env python3
"""
Sentiment Analysis using NLTK's VADER
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool.
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER.

    Returns:
        dict: Compound, positive, negative, neutral scores
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores

def interpret_vader_scores(scores):
    """
    Interpret VADER scores into sentiment label.
    """
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def main():
    # Sample texts for analysis (including some with emojis and slang)
    texts = [
        "I love this product!",
        "This is terrible.",
        "The weather is okay.",
        "I'm extremely happy with the service.",
        "This movie was awful.",
        "It's an average day.",
        "OMG, this is amazing!!!",
        "Not good at all.",
    ]

    print("VADER Sentiment Analysis Results:")
    print("=" * 50)

    for text in texts:
        scores = analyze_sentiment_vader(text)
        label = interpret_vader_scores(scores)
        print(f"Text: {text}")
        print(f"Compound: {scores['compound']:.3f} ({label})")
        print(f"Positive: {scores['pos']:.3f}, Negative: {scores['neg']:.3f}, Neutral: {scores['neu']:.3f}")
        print()

if __name__ == "__main__":
    main()
