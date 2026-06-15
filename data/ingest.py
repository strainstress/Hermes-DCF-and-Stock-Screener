"""Polygon.io data ingestion pipeline.

Pulls fundamentals, price history, and financials from Polygon.io,
caches everything to DuckDB for fast local access.
"""

from pathlib import Path
import time
import requests
import yaml
import duckdb
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

# ── Config ─────────────────────────────────────────────────────

_CONFIG: dict | None = None


def load_config() -> dict:
    """Load config.yaml and overlay .env secrets.

    Returns the merged config dict. Cached after first call.
    """
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path) as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


# ── Polygon Client ─────────────────────────────────────────────


class PolygonClient:
    """Minimal Polygon.io REST API client."""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.BASE_URL}{path}"
        resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()


def get_polygon_client() -> PolygonClient:
    """Return a configured PolygonClient using env POLYGON_API_KEY."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "POLYGON_API_KEY not set. Add it to .env or export it."
        )
    return PolygonClient(api_key)


# ── Data Fetching ──────────────────────────────────────────────


def fetch_fundamentals(ticker: str, client: PolygonClient) -> dict | None:
    """Fetch fundamentals for one ticker from Polygon.io.

    Returns a dict with keys: ticker, market_cap, revenue_ttm, eps_ttm,
    gross_margin, fcf_ttm, roe, debt_equity, current_ratio, and more.
    Returns None on any failure.
    """
    try:
        # Ticker details
        details = client._get(f"/v3/reference/tickers/{ticker}")
        results = details.get("results", {})
        market_cap = results.get("market_cap", 0) or 0

        # Financials (latest annual)
        fin = client._get(
            f"/vX/reference/financials",
            params={"ticker": ticker, "limit": 1, "timeframe": "annual"},
        )
        fin_results = fin.get("results", [])

        if not fin_results:
            logger.warning(f"No financials for {ticker}")
            return None

        fdata = fin_results[0].get("financials", {})
        income = fdata.get("income_statement", {})
        balance = fdata.get("balance_sheet", {})
        cashflow = fdata.get("cash_flow_statement", {})

        def _val(section: dict, key: str) -> float:
            return section.get(key, {}).get("value", 0) or 0

        revenue = _val(income, "revenues")
        gross_profit = _val(income, "gross_profit")
        opex = _val(income, "operating_expenses")
        op_income = _val(income, "operating_income_loss")
        net_income = _val(income, "net_income_loss")
        eps = _val(income, "diluted_earnings_per_share")
        interest_expense = abs(_val(income, "interest_expense_operating"))
        tax_expense = abs(_val(income, "income_tax_expense_benefit"))

        assets = _val(balance, "assets")
        current_assets = _val(balance, "current_assets")
        liabilities = _val(balance, "liabilities")
        current_liabilities = _val(balance, "current_liabilities")
        equity = _val(balance, "equity")
        long_term_debt = _val(balance, "long_term_debt")
        total_debt = long_term_debt  # Polygon doesn't split short-term debt in vX

        op_cf = _val(cashflow, "net_cash_flow_from_operating_activities")
        capex = abs(_val(cashflow, "net_cash_flow_from_investing_activities"))

        # Derived metrics
        gross_margin = gross_profit / revenue if revenue else 0
        fcf = op_cf - capex
        fcf_margin = fcf / revenue if revenue else 0
        roe = net_income / equity if equity else 0
        debt_equity = total_debt / equity if equity else 0
        current_ratio = current_assets / current_liabilities if current_liabilities else 0

        # Altman Z-Score (for non-financials)
        # Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MktCap/TL) + 1.0*(Sales/TA)
        wc = current_assets - current_liabilities
        ebit = op_income + interest_expense  # approximate
        z_numer = (
            1.2 * (wc / assets if assets else 0)
            + 1.4 * (0)  # RE not available from single-year Polygon data
            + 3.3 * (ebit / assets if assets else 0)
            + 0.6 * (market_cap / liabilities if liabilities else 0)
            + 1.0 * (revenue / assets if assets else 0)
        )

        # ROIC = NOPAT / Invested Capital
        # NOPAT = EBIT * (1 - tax_rate)
        tax_rate = tax_expense / (net_income + tax_expense) if (net_income + tax_expense) else 0.21
        nopat = ebit * (1 - tax_rate)
        invested_capital = equity + total_debt
        roic = nopat / invested_capital if invested_capital else 0

        return {
            "ticker": ticker.upper(),
            "name": results.get("name", ""),
            "market_cap": market_cap,
            "revenue_ttm": revenue,
            "eps_ttm": eps,
            "gross_margin": gross_margin,
            "fcf_ttm": fcf,
            "fcf_margin": fcf_margin,
            "roe": roe,
            "roic": roic,
            "debt_equity": debt_equity,
            "current_ratio": current_ratio,
            "altman_z": z_numer,
            "interest_coverage": ebit / interest_expense if interest_expense else 99,
            "shares_outstanding": results.get("weighted_shares_outstanding", 0) or 0,
            "sector": results.get("sic_description", ""),
        }

    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
        return None


def fetch_price_history(
    ticker: str, client: PolygonClient, days: int = 365
) -> list[dict]:
    """Fetch daily OHLCV aggregates from Polygon.

    Returns a list of dicts with keys: date, open, high, low, close, volume.
    """
    try:
        from datetime import datetime, timedelta

        end = datetime.now()
        start = end - timedelta(days=days)

        all_results = []
        # Polygon paginates — get up to 2 years of data
        for _ in range(3):  # max 3 pages (50k results each = 150k bars)
            data = client._get(
                f"/v2/aggs/ticker/{ticker}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
                params={"adjusted": "true", "sort": "asc", "limit": 50000},
            )
            results = data.get("results", [])
            all_results.extend(results)
            if len(results) < 50000:
                break
            # Update start to continue pagination
            if results:
                last_ts = results[-1]["t"] / 1_000_000_000  # nanos → seconds
                start = datetime.fromtimestamp(last_ts) + timedelta(days=1)

        return [
            {
                "date": r["t"],
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"],
                "volume": r["v"],
            }
            for r in all_results
        ]
    except Exception as e:
        logger.error(f"Failed to fetch price history for {ticker}: {e}")
        return []


def fetch_financials(
    ticker: str, client: PolygonClient
) -> list[dict]:
    """Fetch quarterly financial statements for multi-period calculations."""
    try:
        data = client._get(
            f"/vX/reference/financials",
            params={"ticker": ticker, "limit": 20, "timeframe": "quarterly"},
        )
        return data.get("results", [])
    except Exception as e:
        logger.error(f"Failed to fetch financials for {ticker}: {e}")
        return []


# ── Caching ────────────────────────────────────────────────────


def cache_fundamentals(
    ticker: str, data: dict, conn: duckdb.DuckDBPyConnection
) -> None:
    """Write fundamentals data to the DuckDB cache.

    Creates the fundamentals table on first call (upsert pattern).
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker VARCHAR PRIMARY KEY,
            name VARCHAR,
            market_cap DOUBLE,
            revenue_ttm DOUBLE,
            eps_ttm DOUBLE,
            gross_margin DOUBLE,
            fcf_ttm DOUBLE,
            fcf_margin DOUBLE,
            roe DOUBLE,
            roic DOUBLE,
            debt_equity DOUBLE,
            current_ratio DOUBLE,
            altman_z DOUBLE,
            interest_coverage DOUBLE,
            shares_outstanding DOUBLE,
            sector VARCHAR,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute(
        """
        INSERT OR REPLACE INTO fundamentals
            (ticker, name, market_cap, revenue_ttm, eps_ttm, gross_margin,
             fcf_ttm, fcf_margin, roe, roic, debt_equity, current_ratio,
             altman_z, interest_coverage, shares_outstanding, sector)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            data["ticker"],
            data.get("name", ""),
            data.get("market_cap", 0),
            data.get("revenue_ttm", 0),
            data.get("eps_ttm", 0),
            data.get("gross_margin", 0),
            data.get("fcf_ttm", 0),
            data.get("fcf_margin", 0),
            data.get("roe", 0),
            data.get("roic", 0),
            data.get("debt_equity", 0),
            data.get("current_ratio", 0),
            data.get("altman_z", 0),
            data.get("interest_coverage", 0),
            data.get("shares_outstanding", 0),
            data.get("sector", ""),
        ],
    )


def cache_price_history(
    ticker: str, bars: list[dict], conn: duckdb.DuckDBPyConnection
) -> None:
    """Write price bars to DuckDB."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            ticker VARCHAR,
            date BIGINT,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            PRIMARY KEY (ticker, date)
        )
    """)
    for bar in bars:
        conn.execute(
            """
            INSERT OR REPLACE INTO price_history
                (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ticker.upper(),
                bar["date"],
                bar["open"],
                bar["high"],
                bar["low"],
                bar["close"],
                bar["volume"],
            ],
        )


# ── Orchestration ──────────────────────────────────────────────


def ingest_universe(
    tickers: list[str],
    force_refresh: bool = False,
    rate_limit_delay: float = 12.0,  # Polygon free tier: 5 req/min
) -> int:
    """Pull and cache fundamentals for all tickers.

    Args:
        tickers: List of ticker symbols.
        force_refresh: If True, refetch even if cached recently.
        rate_limit_delay: Seconds between API calls (free tier = 12s).

    Returns:
        Number of tickers successfully cached.
    """
    from data.cache import get_cache

    client = get_polygon_client()
    conn = get_cache()

    # Check TTL if not forcing refresh
    if not force_refresh:
        try:
            row = conn.execute(
                "SELECT MAX(fetched_at) FROM fundamentals"
            ).fetchone()
            if row and row[0]:
                last_fetch = row[0]
                # DuckDB returns datetime
                age_hours = (duckdb.execute(
                    "SELECT date_diff('hour', ?::TIMESTAMP, CURRENT_TIMESTAMP)",
                    [last_fetch],
                ).fetchone())[0]
                ttl = load_config().get("data", {}).get("polygon", {}).get("cache_ttl_hours", 24)
                if age_hours < ttl:
                    logger.info(f"Cache is {age_hours}h old (< {ttl}h TTL) — skipping refresh")
                    return len(tickers)
        except Exception:
            pass  # No cache yet — proceed

    cached = 0
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Fetching {ticker}...")
        data = fetch_fundamentals(ticker, client)
        if data:
            cache_fundamentals(ticker, data, conn)
            cached += 1
        else:
            logger.warning(f"Skipped {ticker} — no data returned")

        # Rate limiting
        if i < len(tickers) - 1:
            time.sleep(rate_limit_delay)

    logger.info(f"Cached {cached}/{len(tickers)} tickers")
    return cached
