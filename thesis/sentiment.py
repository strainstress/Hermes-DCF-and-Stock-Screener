"""Multi-source market sentiment fetcher.

Pulls ticker sentiment from:
- Reddit (r/wallstreetbets, r/stocks, r/investing) — free, no API key
- Yahoo Finance news — free via yfinance
- StockTwits — free API

Returns structured sentiment data for injection into thesis prompts
and screening model.
"""

import time
import requests
from datetime import datetime, timedelta
from loguru import logger

# ── Reddit ─────────────────────────────────────────────────────


def fetch_reddit_mentions(
    ticker: str,
    subreddits: list[str] | None = None,
    limit: int = 25,
) -> dict:
    """Fetch recent mentions of a ticker from Reddit.

    Uses Reddit's JSON API (no auth required for read-only).

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        subreddits: List of subreddits to search. Default: wallstreetbets, stocks, investing.
        limit: Max posts to return per subreddit.

    Returns:
        Dict with keys: posts (list of {title, score, num_comments, url, created_utc}),
        total_mentions, avg_score.
    """
    if subreddits is None:
        subreddits = ["wallstreetbets", "stocks", "investing"]

    all_posts = []
    headers = {"User-Agent": "stock-screener/0.1.0"}

    for sub in subreddits:
        try:
            url = f"https://www.reddit.com/r/{sub}/search.json"
            params = {"q": ticker, "limit": limit, "sort": "new", "restrict_sr": "on"}
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            for child in data.get("data", {}).get("children", []):
                post = child["data"]
                all_posts.append({
                    "title": post.get("title", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "created_utc": post.get("created_utc", 0),
                    "subreddit": sub,
                    "source": "reddit",
                })
        except Exception as e:
            logger.debug(f"Reddit r/{sub} failed for {ticker}: {e}")
            continue

    total_mentions = len(all_posts)
    avg_score = (
        sum(p["score"] for p in all_posts) / total_mentions
        if total_mentions > 0
        else 0
    )

    return {
        "posts": sorted(all_posts, key=lambda p: p["score"], reverse=True),
        "total_mentions": total_mentions,
        "avg_score": avg_score,
        "source": "reddit",
    }


# ── Yahoo Finance News ─────────────────────────────────────────


def fetch_yahoo_news(ticker: str, limit: int = 10) -> dict:
    """Fetch recent news articles for a ticker from Yahoo Finance.

    Uses yfinance under the hood.

    Returns:
        Dict with keys: articles (list of {title, publisher, link, published}),
        total_articles.
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        news = stock.news[:limit] if stock.news else []

        articles = []
        for item in news:
            content = item.get("content", {})
            articles.append({
                "title": content.get("title", ""),
                "summary": content.get("summary", "")[:300],
                "publisher": content.get("provider", {}).get("displayName", ""),
                "link": content.get("canonicalUrl", {}).get("url", ""),
                "published": content.get("pubDate", ""),
                "source": "yahoo_finance",
            })

        return {
            "articles": articles,
            "total_articles": len(articles),
            "source": "yahoo_finance",
        }
    except Exception as e:
        logger.debug(f"Yahoo Finance news failed for {ticker}: {e}")
        return {"articles": [], "total_articles": 0, "source": "yahoo_finance"}


# ── StockTwits ─────────────────────────────────────────────────


def fetch_stocktwits(ticker: str, limit: int = 30) -> dict:
    """Fetch recent messages for a ticker from StockTwits.

    Uses StockTwits public API (no key required for recent messages).

    Returns:
        Dict with keys: messages (list of {body, sentiment, created_at}),
        total_messages, sentiment_breakdown.
    """
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        params = {"limit": limit}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        messages = []
        sentiment_counts = {"Bullish": 0, "Bearish": 0, "Neutral": 0}

        for msg in data.get("messages", []):
            sentiment = msg.get("entities", {}).get("sentiment", {})
            sent_label = (
                sentiment.get("basic", "Neutral").capitalize()
                if sentiment
                else "Neutral"
            )

            messages.append({
                "body": msg.get("body", "")[:300],
                "sentiment": sent_label,
                "created_at": msg.get("created_at", ""),
                "user_followers": msg.get("user", {}).get("followers", 0),
                "source": "stocktwits",
            })

            sentiment_counts[sent_label] = sentiment_counts.get(sent_label, 0) + 1

        return {
            "messages": messages,
            "total_messages": len(messages),
            "sentiment_breakdown": sentiment_counts,
            "source": "stocktwits",
        }
    except Exception as e:
        logger.debug(f"StockTwits failed for {ticker}: {e}")
        return {
            "messages": [],
            "total_messages": 0,
            "sentiment_breakdown": {"Bullish": 0, "Bearish": 0, "Neutral": 0},
            "source": "stocktwits",
        }


# ── Unified Fetcher ────────────────────────────────────────────


def fetch_all_sentiment(
    ticker: str,
    include_reddit: bool = True,
    include_yahoo: bool = True,
    include_stocktwits: bool = True,
) -> dict:
    """Fetch sentiment from all available sources for a ticker.

    Returns a unified dict with per-source data and aggregate metrics.
    """
    result = {
        "ticker": ticker.upper(),
        "reddit": None,
        "yahoo_news": None,
        "stocktwits": None,
        "aggregate": {},
        "fetched_at": datetime.now().isoformat(),
    }

    if include_reddit:
        result["reddit"] = fetch_reddit_mentions(ticker)

    if include_yahoo:
        result["yahoo_news"] = fetch_yahoo_news(ticker)

    if include_stocktwits:
        result["stocktwits"] = fetch_stocktwits(ticker)

    # ── Aggregate metrics ──────────────────────────────────────
    total_mentions = 0
    bullish_signals = 0
    bearish_signals = 0
    all_text = []

    # Reddit — use score as proxy for sentiment
    if result["reddit"]:
        total_mentions += result["reddit"]["total_mentions"]
        for p in result["reddit"].get("posts", []):
            all_text.append(p["title"])
            if p["score"] > 10:
                bullish_signals += 1
            elif p["score"] < 0:
                bearish_signals += 1

    # StockTwits — explicit sentiment labels
    if result["stocktwits"]:
        total_mentions += result["stocktwits"]["total_messages"]
        breakdown = result["stocktwits"].get("sentiment_breakdown", {})
        bullish_signals += breakdown.get("Bullish", 0)
        bearish_signals += breakdown.get("Bearish", 0)
        for m in result["stocktwits"].get("messages", []):
            all_text.append(m["body"])

    # Yahoo Finance news titles
    if result["yahoo_news"]:
        for a in result["yahoo_news"].get("articles", []):
            all_text.append(a["title"])
            all_text.append(a.get("summary", ""))

    # Sentiment ratio: 0 = all bearish, 0.5 = neutral, 1.0 = all bullish
    total_signals = bullish_signals + bearish_signals
    sentiment_ratio = (
        bullish_signals / total_signals if total_signals > 0 else 0.5
    )

    result["aggregate"] = {
        "total_mentions": total_mentions,
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
        "sentiment_ratio": round(sentiment_ratio, 3),
        "combined_text": " ".join(all_text)[:5000],  # for NLP scoring
    }

    return result


def sentiment_to_context(sentiment_data: dict) -> str:
    """Convert sentiment data into a text block for the thesis prompt.

    Args:
        sentiment_data: Output from fetch_all_sentiment().

    Returns:
        Markdown-formatted string for inclusion in the LLM prompt.
    """
    agg = sentiment_data.get("aggregate", {})
    tw = sentiment_data.get("stocktwits", {})
    reddit = sentiment_data.get("reddit", {})
    yahoo = sentiment_data.get("yahoo_news", {})

    lines = ["## Recent Market Sentiment\n"]

    # Aggregate
    ratio = agg.get("sentiment_ratio", 0.5)
    if ratio > 0.6:
        sentiment_label = "🟢 Bullish"
    elif ratio < 0.4:
        sentiment_label = "🔴 Bearish"
    else:
        sentiment_label = "🟡 Neutral"

    lines.append(
        f"**Overall Sentiment:** {sentiment_label} "
        f"(ratio: {ratio:.2f}, {agg.get('total_mentions', 0)} mentions)\n"
    )

    # StockTwits
    if tw and tw.get("total_messages", 0) > 0:
        bd = tw.get("sentiment_breakdown", {})
        lines.append(
            f"**StockTwits:** {tw['total_messages']} messages — "
            f"🐂 {bd.get('Bullish', 0)} Bullish, "
            f"🐻 {bd.get('Bearish', 0)} Bearish\n"
        )

    # Reddit
    if reddit and reddit.get("total_mentions", 0) > 0:
        lines.append(
            f"**Reddit:** {reddit['total_mentions']} mentions, "
            f"avg score: {reddit.get('avg_score', 0):.0f}\n"
        )
        for p in reddit.get("posts", [])[:5]:
            lines.append(f"- [{p['score']}] {p['title'][:120]} (r/{p['subreddit']})\n")

    # Yahoo News
    if yahoo and yahoo.get("total_articles", 0) > 0:
        lines.append(f"\n**Recent News Headlines:**\n")
        for a in yahoo.get("articles", [])[:5]:
            lines.append(f"- {a['title'][:150]}\n")

    return "".join(lines)
