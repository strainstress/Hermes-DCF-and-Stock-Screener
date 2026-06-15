"""AI thesis generation orchestrator.

For each shortlist ticker:
1. Fetch source documents (10-K, 10-Q, transcripts, news)
2. Build the prompt
3. Call Claude Sonnet API
4. Validate output against schema
5. Retry once on schema failure
6. Log token usage and cost
"""

import os
import json
import time
from typing import Optional
from loguru import logger
from anthropic import Anthropic, APIStatusError

from thesis.schema import validate_report, AnalystReport
from thesis.prompt import build_messages
from thesis.sources import (
    ticker_to_cik,
    fetch_latest_10k,
    fetch_latest_10q,
    extract_relevant_excerpts,
)


def get_client() -> Anthropic:
    """Return an Anthropic client configured from env."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to .env or export it.")
    return Anthropic(api_key=api_key)


def generate_thesis(
    ticker: str,
    name: str = "",
    sector: str = "",
    market_cap: float = 0,
    description: str = "",
    growth_score: float = 0,
    quality_score: float = 0,
    momentum_score: float = 0,
    composite_score: float = 0,
    sharpe_ratio_val: float = 0,
    hmm_regime: str = "Unknown",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    max_retries: int = 1,
) -> dict:
    """Generate an AI analyst thesis for one ticker.

    Args:
        ticker: Stock ticker symbol.
        All other args: Company context and screening scores.
        model: Anthropic model ID.
        max_tokens: Max output tokens.
        max_retries: Number of retries on schema validation failure.

    Returns:
        Dict with keys: report (AnalystReport as dict), cost, tokens, success, error.
    """
    result = {
        "ticker": ticker,
        "report": None,
        "cost": 0.0,
        "tokens": {"input": 0, "output": 0},
        "success": False,
        "error": "",
    }

    # ── Fetch source documents ────────────────────────────────
    logger.info(f"[{ticker}] Fetching SEC filings...")
    cik = ticker_to_cik(ticker)
    ten_k_text = ""
    ten_q_text = ""

    if cik:
        ten_k_raw = fetch_latest_10k(cik)
        ten_k_text = extract_relevant_excerpts(ten_k_raw, max_chars=12_000)
        ten_q_raw = fetch_latest_10q(cik)
        ten_q_text = extract_relevant_excerpts(ten_q_raw, max_chars=8_000)
        logger.info(
            f"[{ticker}] 10-K: {len(ten_k_text)} chars, 10-Q: {len(ten_q_text)} chars"
        )
    else:
        logger.warning(f"[{ticker}] No CIK found — skipping SEC filings")

    # ── Build prompt ───────────────────────────────────────────
    messages = build_messages(
        ticker=ticker,
        name=name or ticker,
        sector=sector or "Unknown",
        market_cap=market_cap,
        description=description or f"{ticker} — no description available.",
        ten_k_excerpt=ten_k_text,
        ten_q_excerpt=ten_q_text,
        transcripts="",  # Future: FMP integration
        recent_news="",  # Future: NewsAPI integration
        peer_financials="",  # Future: peer comparison
        growth_score=growth_score,
        quality_score=quality_score,
        momentum_score=momentum_score,
        composite_score=composite_score,
        sharpe_ratio=sharpe_ratio_val,
        hmm_regime=hmm_regime,
    )

    # ── Call LLM ───────────────────────────────────────────────
    client = get_client()

    for attempt in range(max_retries + 1):
        try:
            logger.info(
                f"[{ticker}] Calling {model} (attempt {attempt + 1}/{max_retries + 1})..."
            )
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=messages[0]["content"],
                messages=[{"role": "user", "content": messages[1]["content"]}],
            )

            # Track token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            result["tokens"]["input"] = input_tokens
            result["tokens"]["output"] = output_tokens

            # Cost estimate (Sonnet pricing: $3/1M input, $15/1M output)
            cost = (input_tokens / 1_000_000) * 3.0 + (output_tokens / 1_000_000) * 15.0
            result["cost"] = round(cost, 4)

            raw_text = response.content[0].text

            # ── Validate ───────────────────────────────────────
            valid, report, err = validate_report(raw_text)
            if valid and report:
                result["report"] = report.model_dump()
                result["success"] = True
                logger.info(
                    f"[{ticker}] ✓ Thesis generated ({input_tokens}+{output_tokens} tokens, ${cost:.3f})"
                )
                return result
            else:
                logger.warning(f"[{ticker}] Schema validation failed: {err}")
                if attempt < max_retries:
                    # Append error to context and retry
                    messages[1]["content"] += (
                        f"\n\nYour previous output failed JSON schema validation. "
                        f"Error: {err}\n\nPlease correct your output and return valid JSON."
                    )
                else:
                    result["error"] = f"Schema validation failed after {max_retries + 1} attempts: {err}"

        except APIStatusError as e:
            logger.error(f"[{ticker}] API error: {e}")
            result["error"] = str(e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                break
        except Exception as e:
            logger.error(f"[{ticker}] Unexpected error: {e}")
            result["error"] = str(e)
            break

    return result


def generate_batch(
    tickers: list[dict],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    rate_limit_delay: float = 1.0,
) -> list[dict]:
    """Generate theses for a batch of tickers.

    Args:
        tickers: List of dicts with keys matching generate_thesis() params.
        model: Anthropic model ID.
        max_tokens: Max output tokens per call.
        rate_limit_delay: Seconds between API calls.

    Returns:
        List of result dicts, one per ticker.
    """
    results = []
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for i, t in enumerate(tickers):
        logger.info(f"--- [{i+1}/{len(tickers)}] {t.get('ticker', '???')} ---")
        result = generate_thesis(
            ticker=t.get("ticker", ""),
            name=t.get("name", ""),
            sector=t.get("sector", ""),
            market_cap=t.get("market_cap", 0),
            description=t.get("description", ""),
            growth_score=t.get("growth_score", 0),
            quality_score=t.get("quality_score", 0),
            momentum_score=t.get("momentum_score", 0),
            composite_score=t.get("composite_score", 0),
            sharpe_ratio_val=t.get("sharpe_ratio", 0),
            hmm_regime=t.get("hmm_regime", "Unknown"),
            model=model,
            max_tokens=max_tokens,
        )
        results.append(result)
        total_cost += result["cost"]
        total_input_tokens += result["tokens"]["input"]
        total_output_tokens += result["tokens"]["output"]

        if i < len(tickers) - 1:
            time.sleep(rate_limit_delay)

    n_success = sum(1 for r in results if r["success"])
    logger.info(
        f"Batch complete: {n_success}/{len(tickers)} successful. "
        f"Tokens: {total_input_tokens}+{total_output_tokens}. "
        f"Cost: ${total_cost:.3f}"
    )
    return results
