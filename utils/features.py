
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return analyzer.polarity_scores(text)

def extract_features(text, score):

    text = str(text)

    sentiment = get_sentiment(text)

    word_count = len(text.split())

    sentence_count = len(
        re.split(r'[.!?]+', text)
    )

    uppercase_ratio = (
        sum(1 for c in text if c.isupper()) /
        max(len(text), 1)
    )

    coherence = 0

    if score >= 4 and sentiment['compound'] > 0:
        coherence = 1

    elif score <= 2 and sentiment['compound'] < 0:
        coherence = 1

    return {
        "word_count": word_count,
        "char_count": len(text),
        "sentence_count": sentence_count,
        "uppercase_ratio": uppercase_ratio,
        "sentiment_compound": sentiment['compound'],
        "sentiment_positive": sentiment['pos'],
        "sentiment_negative": sentiment['neg'],
        "coherence": coherence
    }
