# Sentiment Analysis

This repository provides a simple introduction to sentiment analysis using Python. Sentiment analysis is the process of determining whether a piece of writing is positive, negative, or neutral. It's a common task in natural language processing (NLP) and can be applied to customer reviews, social media posts, and more.

## Prerequisites

- Python 3.6 or higher
- Required libraries: Install them using `pip install -r requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ronxldwilson/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Lexicon-Based (TextBlob)
```
python sentiment_analysis.py
```
Analyzes sample texts using TextBlob's polarity and subjectivity.

### Lexicon-Based (VADER)
```
python vader_sentiment.py
```
Uses NLTK's VADER for social media-friendly sentiment analysis.

### Machine Learning (SVM with TF-IDF)
```
python ml_sentiment.py
```
Trains a Support Vector Machine on movie reviews (achieves ~85% accuracy).

### Deep Learning (LSTM)
```
python lstm_sentiment.py
```
Trains an LSTM neural network for advanced sentiment classification.

## Example Output

```
Text: I love this product!
Polarity: 0.5 (Positive)
Subjectivity: 0.6

Text: This is terrible.
Polarity: -1.0 (Negative)
Subjectivity: 1.0

Text: The weather is okay.
Polarity: 0.5 (Neutral)
Subjectivity: 0.5
```

## How It Works

The script uses the TextBlob library, which provides a simple API for sentiment analysis. It calculates:
- **Polarity**: A float from -1.0 to 1.0, where -1.0 is very negative, 0 is neutral, and 1.0 is very positive.
- **Subjectivity**: A float from 0.0 to 1.0, where 0.0 is very objective and 1.0 is very subjective.

## Other Sentiment Analysis Methods

Sentiment analysis can be approached in several ways, each with its strengths:

### 1. Lexicon-Based Methods
- **Description**: Use predefined dictionaries of words with associated sentiment scores. No training data required.
- **Examples**:
  - VADER (Valence Aware Dictionary and sEntiment Reasoner) - Good for social media text
  - SentiWordNet - Based on WordNet
- **Pros**: Fast, no training needed
- **Cons**: Doesn't understand context or sarcasm well
- **Try it**: Run `python vader_sentiment.py` for VADER example

### 2. Machine Learning Methods
- **Description**: Train classifiers on labeled datasets using traditional ML algorithms.
- **Examples**:
  - Naive Bayes
  - Support Vector Machines (SVM)
  - Logistic Regression
- **Pros**: Can learn complex patterns
- **Cons**: Requires labeled training data
- **Libraries**: Scikit-learn, NLTK

### 3. Deep Learning Methods
- **Description**: Use neural networks to learn representations and sentiments.
- **Examples**:
  - Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM)
  - Transformer models like BERT, GPT
- **Pros**: State-of-the-art accuracy, handles context well
- **Cons**: Requires large amounts of data and computational resources
- **Libraries**: TensorFlow, PyTorch, Hugging Face Transformers

### 4. Hybrid Methods
- **Description**: Combine lexicon-based and ML approaches for better results.
- **Examples**: Use lexicon for initial scoring, then ML for refinement

### 5. Aspect-Based Sentiment Analysis
- **Description**: Analyze sentiment towards specific aspects or entities in text.
- **Example**: "The food was great but service was slow" - positive for food, negative for service
- **Libraries**: SpaCy, Stanford CoreNLP

## Next Steps

To learn more:
- Experiment with your own texts in the script.
- Try using NLTK or spaCy for more advanced analysis.
- Explore machine learning approaches with datasets like IMDB reviews." 
