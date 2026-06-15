"""Individual factor calculations for the multi-factor scoring model.

Each factor is a pure function: takes data, returns a float.
Factors return raw values — percentile ranking happens in sector_percentiles.py.
"""

import numpy as np
import pandas as pd
from loguru import logger


# ── Growth Factors ─────────────────────────────────────────────


def revenue_yoy_ttm(fundamentals: dict) -> float:
    """Revenue year-over-year growth (TTM vs prior TTM)."""
    return _safe_float(fundamentals.get("revenue_yoy_ttm", 0))


def revenue_cagr_3yr(financials: list[dict]) -> float:
    """3-year revenue CAGR from quarterly financials."""
    try:
        if not financials or len(financials) < 12:  # Need 12 quarters = 3 years
            return 0.0
        revenues = []
        for f in financials[:12]:
            income = f.get("financials", {}).get("income_statement", {})
            rev = income.get("revenues", {}).get("value", 0) or 0
            revenues.append(rev)
        if revenues[0] <= 0:
            return 0.0
        cagr = (revenues[0] / revenues[-1]) ** (1 / 3) - 1 if revenues[-1] > 0 else 0
        return cagr
    except Exception:
        return 0.0


def eps_growth_ttm(fundamentals: dict) -> float:
    """EPS year-over-year growth (TTM)."""
    return _safe_float(fundamentals.get("eps_yoy_ttm", 0))


def forward_revenue_growth(fundamentals: dict) -> float:
    """Consensus forward revenue growth estimate (default 0 if unavailable)."""
    return _safe_float(fundamentals.get("forward_revenue_growth", 0))


def reinvestment_rate(fundamentals: dict) -> float:
    """Reinvestment rate = (capex + R&D) / revenue. Higher = more growth investment."""
    revenue = _safe_float(fundamentals.get("revenue_ttm", 0))
    if revenue <= 0:
        return 0.0
    capex = abs(_safe_float(fundamentals.get("capex_ttm", 0)))
    rd = abs(_safe_float(fundamentals.get("rd_expense_ttm", 0)))
    return (capex + rd) / revenue


def rule_of_40(fundamentals: dict) -> float:
    """Rule of 40 = revenue growth % + FCF margin %."""
    rev_growth = revenue_yoy_ttm(fundamentals)
    fcf_margin = _safe_float(fundamentals.get("fcf_margin", 0))
    return (rev_growth * 100) + (fcf_margin * 100)


# ── Quality Factors ────────────────────────────────────────────


def roe_ttm(fundamentals: dict) -> float:
    """Return on equity (TTM)."""
    return _safe_float(fundamentals.get("roe", 0))


def roic(fundamentals: dict) -> float:
    """Return on invested capital."""
    return _safe_float(fundamentals.get("roic", 0))


def gross_margin_ttm(fundamentals: dict) -> float:
    """Gross margin (TTM)."""
    return _safe_float(fundamentals.get("gross_margin", 0))


def fcf_margin_ttm(fundamentals: dict) -> float:
    """Free cash flow margin (TTM)."""
    return _safe_float(fundamentals.get("fcf_margin", 0))


def net_debt_ebitda(fundamentals: dict) -> float:
    """Net debt / EBITDA. Lower is better. Return raw — scoring inverts it."""
    total_debt = _safe_float(fundamentals.get("total_debt", 0))
    cash = _safe_float(fundamentals.get("cash_and_equivalents", 0))
    ebitda = _safe_float(fundamentals.get("ebitda_ttm", 0))
    if ebitda <= 0:
        return 999.0  # effectively infinite — bad
    return (total_debt - cash) / ebitda


def accruals_ratio(financials: list[dict]) -> float:
    """Accruals ratio = |(Net Income - OpCF)| / Total Assets. Lower = better quality."""
    try:
        if not financials:
            return 0.0
        f = financials[0].get("financials", {})
        income = f.get("income_statement", {})
        cf = f.get("cash_flow_statement", {})
        balance = f.get("balance_sheet", {})
        ni = income.get("net_income_loss", {}).get("value", 0) or 0
        opcf = cf.get("net_cash_flow_from_operating_activities", {}).get("value", 0) or 0
        assets = balance.get("assets", {}).get("value", 1) or 1
        return abs(ni - opcf) / assets
    except Exception:
        return 0.0


def interest_coverage(fundamentals: dict) -> float:
    """EBIT / Interest expense. Higher = safer."""
    return _safe_float(fundamentals.get("interest_coverage", 0))


# ── Momentum Factors ───────────────────────────────────────────


def return_6m_excl_1m(price_history: list[dict]) -> float:
    """6-month price return excluding the most recent month."""
    return _price_return(price_history, start_offset=126, end_offset=21)


def return_12m(price_history: list[dict]) -> float:
    """12-month price return."""
    return _price_return(price_history, start_offset=252, end_offset=0)


def return_1m(price_history: list[dict]) -> float:
    """1-month price return (mean reversion guard)."""
    return _price_return(price_history, start_offset=21, end_offset=0)


def earnings_surprise(fundamentals: dict) -> float:
    """Average earnings surprise over last 4 quarters. Positive = beats estimates."""
    return _safe_float(fundamentals.get("earnings_surprise_avg", 0))


def ma_ratio_50_200(price_history: list[dict]) -> float:
    """50-day MA / 200-day MA ratio (> 1.0 = golden cross, bullish)."""
    closes = _extract_closes(price_history)
    if len(closes) < 200:
        return 1.0  # neutral if insufficient data
    ma50 = np.mean(closes[-50:])
    ma200 = np.mean(closes[-200:])
    return ma50 / ma200 if ma200 > 0 else 1.0


# ── Sharpe Ratio (Risk-Adjusted Momentum) ──────────────────────


def sharpe_ratio(
    price_history: list[dict],
    risk_free_rate: float = 0.045,
    window_days: int = 252,
) -> float:
    """Annualized Sharpe ratio over the lookback window.

    Sharpe = (annualized_return - risk_free_rate) / annualized_volatility.

    Stocks with moderate returns and low vol score HIGHER than
    volatile high-flyers — this surfaces "quiet compounders."
    """
    closes = _extract_closes(price_history)
    if len(closes) < window_days:
        return 0.0

    closes = closes[-window_days:]
    daily_returns = np.diff(closes) / closes[:-1]
    if len(daily_returns) < 20:
        return 0.0

    ann_return = np.mean(daily_returns) * 252
    ann_vol = np.std(daily_returns, ddof=1) * np.sqrt(252)

    if ann_vol <= 0:
        return 0.0

    return (ann_return - risk_free_rate) / ann_vol


# ── Helpers ────────────────────────────────────────────────────


def _safe_float(val) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    try:
        return float(val or 0)
    except (TypeError, ValueError):
        return 0.0


def _extract_closes(price_history: list[dict]) -> np.ndarray:
    """Extract close prices from price bars, sorted by date."""
    if not price_history:
        return np.array([])
    sorted_bars = sorted(price_history, key=lambda b: b.get("date", 0))
    return np.array([b.get("close", 0) or 0 for b in sorted_bars], dtype=float)


def _price_return(
    price_history: list[dict],
    start_offset: int = 252,
    end_offset: int = 0,
) -> float:
    """Calculate price return from start_offset days ago to end_offset days ago."""
    closes = _extract_closes(price_history)
    if len(closes) < start_offset:
        return 0.0
    start_price = closes[-start_offset] if start_offset > 0 else closes[0]
    end_price = closes[-end_offset] if end_offset > 0 else closes[-1]
    if start_price <= 0:
        return 0.0
    return (end_price / start_price) - 1


def compute_all_factors(
    fundamentals: dict,
    financials: list[dict] | None = None,
    price_history: list[dict] | None = None,
    config: dict | None = None,
) -> dict[str, float]:
    """Compute all growth, quality, and momentum factors for one ticker.

    Args:
        fundamentals: Dict from fetch_fundamentals().
        financials: Optional quarterly financials list.
        price_history: Optional daily price bars.
        config: Optional config dict (for Sharpe/HMM settings).

    Returns:
        Dict mapping factor_name → raw_value.
    """
    financials = financials or []
    price_history = price_history or []
    cfg = config or {}

    sharpe_cfg = cfg.get("scoring", {}).get("sharpe", {})
    risk_free = sharpe_cfg.get("risk_free_rate", 0.045)
    sharpe_window = sharpe_cfg.get("window_days", 252)

    return {
        # Growth
        "revenue_yoy_ttm": revenue_yoy_ttm(fundamentals),
        "revenue_cagr_3yr": revenue_cagr_3yr(financials),
        "eps_growth_ttm": eps_growth_ttm(fundamentals),
        "forward_revenue_growth": forward_revenue_growth(fundamentals),
        "reinvestment_rate": reinvestment_rate(fundamentals),
        "rule_of_40": rule_of_40(fundamentals),
        # Quality
        "roe_ttm": roe_ttm(fundamentals),
        "roic": roic(fundamentals),
        "gross_margin_ttm": gross_margin_ttm(fundamentals),
        "fcf_margin_ttm": fcf_margin_ttm(fundamentals),
        "net_debt_ebitda": net_debt_ebitda(fundamentals),
        "accruals_ratio": accruals_ratio(financials),
        "interest_coverage": interest_coverage(fundamentals),
        # Momentum
        "return_6m_excl_1m": return_6m_excl_1m(price_history),
        "return_12m": return_12m(price_history),
        "return_1m": return_1m(price_history),
        "earnings_surprise": earnings_surprise(fundamentals),
        "ma_ratio_50_200": ma_ratio_50_200(price_history),
        # Risk-adjusted
        "sharpe_ratio": sharpe_ratio(price_history, risk_free, sharpe_window),
    }
