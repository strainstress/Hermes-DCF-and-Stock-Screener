"""NLP Sentiment scoring for market sentiment data.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) from NLTK —
optimized for social media and short-form financial text.

Feeds into the screening model as a "Sentiment" subscore.
"""

import numpy as np
from loguru import logger

# Lazy-load VADER — installs lexicon on first use
_vader = None


def _get_vader():
    """Get or initialize VADER sentiment analyzer."""
    global _vader
    if _vader is None:
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
        except Exception:
            pass  # will use fallback
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("nltk not installed — sentiment scoring disabled")
            _vader = None
    return _vader


def score_text_sentiment(text: str) -> dict:
    """Score the sentiment of a text using VADER.

    Args:
        text: Text to analyze (titles, posts, news combined).

    Returns:
        Dict with keys: compound (float -1 to 1), pos, neu, neg, label.
    """
    if not text or not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "neutral"}

    analyzer = _get_vader()
    if analyzer is None:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "neutral"}

    scores = analyzer.polarity_scores(text[:5000])  # VADER works on short texts

    if scores["compound"] >= 0.05:
        label = "positive"
    elif scores["compound"] <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "compound": round(scores["compound"], 4),
        "pos": round(scores["pos"], 4),
        "neu": round(scores["neu"], 4),
        "neg": round(scores["neg"], 4),
        "label": label,
    }


def compute_sentiment_score(sentiment_data: dict) -> float:
    """Compute a composite sentiment score from multi-source data.

    Combines:
    - VADER compound score on combined text (40% weight)
    - StockTwits bull/bear ratio (35% weight)
    - Reddit engagement score (25% weight)

    Returns a score 0-100 where higher = more bullish/buzz.
    """
    agg = sentiment_data.get("aggregate", {})
    combined_text = agg.get("combined_text", "")

    # VADER score
    vader = score_text_sentiment(combined_text)
    vader_component = (vader["compound"] + 1) / 2 * 100  # scale -1..1 → 0..100

    # StockTwits ratio
    st_ratio = agg.get("sentiment_ratio", 0.5)
    st_component = st_ratio * 100

    # Reddit engagement (log-scaled avg score)
    reddit_data = sentiment_data.get("reddit", {})
    reddit_avg = reddit_data.get("avg_score", 0)
    reddit_component = min(100, max(0, 50 + np.log1p(max(reddit_avg, 1)) * 10))

    # If no data at all, return neutral
    if agg.get("total_mentions", 0) == 0:
        return 50.0

    # Weighted composite
    score = (
        vader_component * 0.40
        + st_component * 0.35
        + reddit_component * 0.25
    )

    return round(min(100, max(0, score)), 1)


def sentiment_label(score: float) -> str:
    """Convert a sentiment score (0-100) to a human label."""
    if score >= 70:
        return "🟢 Very Bullish"
    elif score >= 55:
        return "🟢 Bullish"
    elif score >= 45:
        return "🟡 Neutral"
    elif score >= 30:
        return "🔴 Bearish"
    else:
        return "🔴 Very Bearish"
