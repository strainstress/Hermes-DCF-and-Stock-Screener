"""LLM prompt templates for AI analyst thesis generation.

Builds structured prompts for Claude Sonnet with:
- System prompt (buy-side analyst persona)
- Company context (ticker, sector, market cap)
- Source materials (10-K, 10-Q, transcripts, news)
- Scoring inputs from Stage 1
- Exact JSON output schema
"""

SYSTEM_PROMPT = """You are a buy-side equity analyst at a long-biased, fundamentals-driven fund. Your job is to produce concise, evidence-backed investment theses.

RULES:
1. Be specific. Cite sources by section, page, or direct quote. Never make generic statements like "this is a great company."
2. If data is missing for a particular angle, say so explicitly — don't invent.
3. Use only the data provided in the source materials below. Do not bring in external knowledge.
4. Your output must be valid JSON matching the exact schema specified.
5. Be intellectually honest. Acknowledge bear cases with the same rigor as bull cases.
"""


def build_user_prompt(
    ticker: str,
    name: str,
    sector: str,
    market_cap: float,
    description: str,
    ten_k_excerpt: str,
    ten_q_excerpt: str,
    transcripts: str,
    recent_news: str,
    peer_financials: str,
    growth_score: float,
    quality_score: float,
    momentum_score: float,
    composite_score: float,
    sharpe_ratio: float,
    hmm_regime: str,
    sentiment_context: str = "",
) -> str:
    """Build the full user prompt for one ticker.

    Returns a string ready to send to Claude.
    """
    prompt = f"""## Company
- **Ticker:** {ticker}
- **Name:** {name}
- **Sector:** {sector}
- **Market Cap:** ${market_cap:,.0f}
- **Description:** {description}

## Stage 1 Screening Scores (0-100 percentile within sector)
- Growth: {growth_score:.1f}
- Quality: {quality_score:.1f}
- Momentum: {momentum_score:.1f}
- Composite: {composite_score:.1f}
- Sharpe Ratio: {sharpe_ratio:.3f}
- Current Regime: {hmm_regime}

## Source Materials

### Most Recent 10-K (excerpts)
{ten_k_excerpt or "Not available"}

### Most Recent 10-Q (excerpts)
{ten_q_excerpt or "Not available"}

### Recent Earnings Call Transcripts
{transcripts or "Not available"}

### Recent News (Last 30 Days)
{recent_news or "Not available"}

### Peer Financial Comparison
{peer_financials or "Not available"}

{sentiment_context}

## Your Task

Produce an investment thesis as a JSON object matching this exact schema:

```json
{{
  "tldr": "2-3 sentence executive summary of the investment thesis",
  "bull_case": ["Point 1", "Point 2", "Point 3"],
  "bear_case": ["Risk 1", "Risk 2", "Risk 3"],
  "key_metrics_to_watch": ["Metric 1", "Metric 2"],
  "qualitative_scores": {{
    "moat": 0-10,
    "management": 0-10,
    "capital_allocation": 0-10,
    "industry_position": 0-10
  }},
  "peer_comparison": "1-2 paragraphs comparing this company to its closest peers on key metrics and positioning",
  "verdict": "Strong Buy | Buy | Watch | Pass | Sell",
  "confidence": 0-100,
  "sources_cited": ["10-K p.42", "Q3 transcript re: guidance", "News: WSJ 6/10/26"]
}}
```

Be concise but substantive. Every claim in the bull/bear case should be traceable to the source materials above.
"""
    return prompt


def build_messages(
    ticker: str,
    name: str,
    sector: str,
    market_cap: float,
    description: str,
    ten_k_excerpt: str = "",
    ten_q_excerpt: str = "",
    transcripts: str = "",
    recent_news: str = "",
    peer_financials: str = "",
    sentiment_context: str = "",
    growth_score: float = 0,
    quality_score: float = 0,
    momentum_score: float = 0,
    composite_score: float = 0,
    sharpe_ratio: float = 0,
    hmm_regime: str = "Unknown",
) -> list[dict]:
    """Build the full messages array for the Anthropic API.

    Returns:
        List of message dicts: [system_message, user_message].
    """
    user_prompt = build_user_prompt(
        ticker=ticker,
        name=name,
        sector=sector,
        market_cap=market_cap,
        description=description,
        ten_k_excerpt=ten_k_excerpt,
        ten_q_excerpt=ten_q_excerpt,
        transcripts=transcripts,
        recent_news=recent_news,
        peer_financials=peer_financials,
        sentiment_context=sentiment_context,
        growth_score=growth_score,
        quality_score=quality_score,
        momentum_score=momentum_score,
        composite_score=composite_score,
        sharpe_ratio=sharpe_ratio,
        hmm_regime=hmm_regime,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
