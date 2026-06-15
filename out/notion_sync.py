"""Notion database sync for the Stock Screener.

Pushes scored tickers + thesis reports to a Notion database.
Each ticker = one row with scores, verdict, thesis.

Requires:
- NOTION_TOKEN in .env
- NOTION_DATABASE_ID in .env (the target database must exist)
- The database must have the right properties (created on first run if missing)
"""

import os
import json
from pathlib import Path
from loguru import logger
from notion_client import Client


def get_notion_client() -> Client:
    """Return a configured Notion client."""
    token = os.getenv("NOTION_TOKEN", "")
    if not token:
        raise RuntimeError("NOTION_TOKEN not set. Add it to .env or export it.")
    return Client(auth=token)


def get_or_create_db_id() -> str:
    """Get the Notion database ID from env."""
    db_id = os.getenv("NOTION_DATABASE_ID", "")
    if not db_id:
        raise RuntimeError(
            "NOTION_DATABASE_ID not set. Create a Notion database, "
            "share it with your integration, and add the ID to .env."
        )
    return db_id


def push_ticker(
    ticker: str,
    name: str = "",
    sector: str = "",
    market_cap: float = 0,
    revenue_ttm: float = 0,
    fcf_margin: float = 0,
    roe: float = 0,
    composite_score: float = 0,
    verdict: str = "",
    tldr: str = "",
    confidence: int = 0,
    hidden_gem_score: float = 0,
    hmm_regime: str = "",
    sharpe_ratio: float = 0,
    moat_score: int = 0,
) -> bool:
    """Push or update a single ticker row in the Notion database.

    Uses the ticker as a unique identifier (stored in a "Ticker" title property).

    Returns True on success.
    """
    client = get_notion_client()
    db_id = get_or_create_db_id()

    # Build properties payload
    properties = {
        "Ticker": {"title": [{"text": {"content": ticker}}]},
        "Company": {"rich_text": [{"text": {"content": name[:100]}}]},
        "Sector": {"select": {"name": sector[:100] if sector else "Unknown"}},
        "Market Cap ($B)": {"number": round(market_cap / 1e9, 2) if market_cap else 0},
        "Revenue TTM ($B)": {"number": round(revenue_ttm / 1e9, 2) if revenue_ttm else 0},
        "FCF Margin": {"number": round(fcf_margin * 100, 1) if fcf_margin else 0},
        "ROE": {"number": round(roe * 100, 1) if roe else 0},
        "Composite Score": {"number": round(composite_score, 1)},
        "Hidden Gem": {"number": round(hidden_gem_score, 1)},
        "Sharpe": {"number": round(sharpe_ratio, 3)},
        "Regime": {"select": {"name": hmm_regime[:50] if hmm_regime else "Unknown"}},
        "Verdict": {"select": {"name": verdict if verdict else "Pending"}},
        "Confidence": {"number": confidence},
        "Moat": {"number": moat_score},
    }

    try:
        # Check if ticker already exists (search by title)
        existing = client.databases.query(
            database_id=db_id,
            filter={
                "property": "Ticker",
                "title": {"equals": ticker},
            },
        )

        if existing.get("results"):
            # Update existing page
            page_id = existing["results"][0]["id"]
            client.pages.update(page_id=page_id, properties=properties)

            # Append thesis as page content if provided
            if tldr:
                client.blocks.children.append(
                    block_id=page_id,
                    children=[
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": tldr[:2000]}}]
                            },
                        }
                    ],
                )
            logger.debug(f"Updated {ticker} in Notion")
        else:
            # Create new page
            client.pages.create(
                parent={"database_id": db_id},
                properties=properties,
                children=(
                    [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": tldr[:2000]}}]
                            },
                        }
                    ]
                    if tldr
                    else []
                ),
            )
            logger.debug(f"Created {ticker} in Notion")

        return True

    except Exception as e:
        logger.error(f"Notion push failed for {ticker}: {e}")
        return False


def sync_from_cache(
    limit: int | None = None,
    verdict_filter: str | None = None,
) -> dict:
    """Sync scored tickers from DuckDB cache to Notion.

    Args:
        limit: Max tickers to sync (None = all).
        verdict_filter: Only sync tickers with this verdict.

    Returns:
        Dict with counts: {synced, skipped, errors}.
    """
    from data.cache import get_cache

    conn = get_cache()
    df = conn.execute("SELECT * FROM fundamentals ORDER BY market_cap DESC").df()

    if df.empty:
        logger.warning("No data in cache — run 'python run.py ingest' first")
        return {"synced": 0, "skipped": 0, "errors": 0}

    # Load any thesis reports
    reports = {}
    reports_dir = Path(__file__).resolve().parent / "reports"
    if reports_dir.exists():
        for f in reports_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    ticker = data.get("ticker", "")
                    if ticker:
                        reports[ticker] = data
            except Exception:
                pass

    synced = 0
    skipped = 0
    errors = 0

    rows = df.iterrows() if limit is None else list(df.head(limit).iterrows())

    for _, row in rows:
        ticker = row["ticker"]

        # Get thesis data if available
        thesis = reports.get(ticker, {}).get("report", {})

        try:
            success = push_ticker(
                ticker=ticker,
                name=row.get("name", ""),
                sector=row.get("sector", ""),
                market_cap=row.get("market_cap", 0),
                revenue_ttm=row.get("revenue_ttm", 0),
                fcf_margin=row.get("fcf_margin", 0),
                roe=row.get("roe", 0),
                composite_score=row.get("composite_score", 0),
                verdict=thesis.get("verdict", ""),
                tldr=thesis.get("tldr", ""),
                confidence=thesis.get("confidence", 0),
                moat_score=thesis.get("qualitative_scores", {}).get("moat", 0),
            )
            if success:
                synced += 1
            else:
                errors += 1
        except Exception as e:
            logger.error(f"Error syncing {ticker}: {e}")
            errors += 1

    logger.info(f"Notion sync: {synced} synced, {skipped} skipped, {errors} errors")
    return {"synced": synced, "skipped": skipped, "errors": errors}
