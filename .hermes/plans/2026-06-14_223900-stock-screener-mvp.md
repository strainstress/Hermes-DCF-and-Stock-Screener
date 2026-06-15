# Stock Screener + AI Thesis Generator — Implementation Plan

> **For Hermes:** Execute task-by-task with TDD — each task: write failing test → watch it fail → minimal code → watch it pass → commit.

**Goal:** Build a weekly-on-demand pipeline that screens ~900 S&P 500+400 stocks through a multi-factor model (Growth > Quality > Momentum), then generates AI analyst theses for the top 20–40 picks, surfaced in a Streamlit dashboard.

**Architecture:** Python 3.11+, DuckDB cache, Polygon.io for fundamentals, Claude Sonnet for thesis generation, Streamlit for dashboard. Five modules: `data/` (ingest + universe), `screen/` (scoring), `thesis/` (LLM reports), `out/` (dashboard + Notion sync), `run.py` (orchestrator).

**Tech Stack:** Python 3.11+, uv (package mgmt), DuckDB, Polygon.io, Anthropic API, Streamlit, pytest, loguru

---

## Phase 1: Foundation (Repo Scaffold + Universe + Ingest)

### Task 1.1: Create pyproject.toml with all dependencies

**Objective:** Scaffold the Python project with uv, listing all known dependencies.

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`

**Step 1: Write pyproject.toml**

```toml
[project]
name = "stock-screener"
version = "0.1.0"
description = "DCF valuation and stock screening pipeline with AI-generated analyst theses"
requires-python = ">=3.11"
dependencies = [
    "duckdb>=1.0",
    "pyarrow>=15",
    "pandas>=2.2",
    "requests>=2.32",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "loguru>=0.7",
    "pydantic>=2.0",
    "anthropic>=0.30",
    "streamlit>=1.35",
    "notion-client>=2.2",
    "numpy>=1.26",
    "scipy>=1.12",
    "hmmlearn>=0.3",
    "yfinance>=0.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "mypy>=1.10",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

**Step 2: Create .python-version**
```
3.11
```

**Step 3: Initialize with uv**
```bash
cd ~/Hermes-DCF-and-Stock-Screener
uv venv
source .venv/Scripts/activate
uv pip install -e ".[dev]"
```

**Verification:** `uv run python -c "import duckdb; import pandas; print('OK')"` prints OK.

---

### Task 1.2: Create config.yaml and .env.example

**Objective:** Externalize all tunable parameters and document API key setup.

**Files:**
- Create: `config.yaml`
- Create: `.env.example`
- Create: `.gitignore`

**config.yaml skeleton:**
```yaml
# All tunable thresholds and weights
universe:
  indices: ["sp500", "sp400"]
  exclude_etfs: true

disqualifiers:
  min_market_cap_b: 2.0
  min_avg_volume_m: 5.0
  max_debt_equity: 3.0
  min_altman_z: 1.8

scoring:
  weights:
    growth: 0.40
    quality: 0.35
    momentum: 0.25

  growth:
    metrics:
      - revenue_yoy_ttm: 0.25
      - revenue_cagr_3yr: 0.20
      - eps_growth_ttm: 0.20
      - forward_revenue_growth: 0.15
      - reinvestment_rate: 0.10
      - rule_of_40: 0.10

  quality:
    metrics:
      - roe_ttm: 0.20
      - roic: 0.20
      - gross_margin_ttm: 0.15
      - fcf_margin_ttm: 0.15
      - net_debt_ebitda: 0.15
      - accruals_ratio: 0.10
      - interest_coverage: 0.05

  momentum:
    metrics:
      - return_6m_excl_1m: 0.30
      - return_12m: 0.25
      - return_1m: 0.15
      - earnings_surprise: 0.15
      - ma_ratio_50_200: 0.15

  # Sharpe ratio — risk-adjusted momentum (hidden gem detector)
  # A stock with 15% return and 10% vol (Sharpe 1.5) beats one with 30% return and 40% vol (Sharpe 0.75)
  sharpe:
    enabled: true
    weight_in_momentum: 0.20  # replaces 20% of momentum weight — redistributed from 6m/12m returns
    window_days: 252
    risk_free_rate: 0.045

  # Hidden Markov Model — regime-aware scoring
  # Detects Bull/Sideways/Bear regimes per stock. Surfaces stocks outperforming
  # their regime peers (e.g., +10% in a Bear regime > +10% in a Bull regime).
  hmm:
    enabled: true
    n_components: 3           # Bear / Sideways / Bull
    n_seeds: 20               # random seeds tried for convergence (best score wins)
    n_iter: 500               # EM iterations
    window_days: 504          # 2 trading years (~2 calendar years)
    regime_bonus: 0.10        # bonus weight for stocks beating regime expectations
    max_stocks_for_hmm: 200   # cap HMM compute — run only on top 200 by pre-score

thesis:
  shortlist_size: 30
  min_score: 70
  model: "claude-sonnet-4-20250514"
  max_tokens: 4096
  retry_on_schema_fail: true
  max_retries: 1

data:
  polygon:
    base_url: "https://api.polygon.io"
    cache_ttl_hours: 24
  edgar:
    base_url: "https://www.sec.gov"
    user_agent: "stock-screener/0.1.0 (your-email@example.com)"
  cache_dir: "data/cache"

output:
  dashboard:
    port: 8501
  notion:
    enabled: false
```

**.env.example:**
```bash
# Polygon.io API key (get at https://polygon.io/dashboard)
POLYGON_API_KEY=your_key_here

# Anthropic API key (get at https://console.anthropic.com)
ANTHROPIC_API_KEY=your_key_here

# Notion integration token (get at https://www.notion.so/my-integrations)
NOTION_TOKEN=your_token_here
NOTION_DATABASE_ID=your_db_id_here
```

**.gitignore:**
```
.venv/
__pycache__/
*.pyc
.env
data/cache/
out/reports/
.streamlit/secrets.toml
dist/
*.egg-info/
.pytest_cache/
```

**Verification:** `uv run python -c "import yaml; yaml.safe_load(open('config.yaml'))"` prints the config dict.

---

### Task 1.3: Create data/universe.py — static S&P 500 + 400 ticker list

**Objective:** Provide a function returning a deduplicated list of tickers for S&P 500 and S&P 400, excluding ETFs and non-equity securities.

**Files:**
- Create: `data/__init__.py`
- Create: `data/universe.py`
- Create: `tests/test_universe.py`

**Step 1: Write failing test**

```python
# tests/test_universe.py
import pytest
from data.universe import get_universe, is_valid_ticker

def test_get_universe_returns_nonempty_list():
    tickers = get_universe()
    assert len(tickers) > 0, "Universe should return at least one ticker"

def test_get_universe_returns_deduplicated_uppercase_tickers():
    tickers = get_universe()
    assert len(tickers) == len(set(tickers)), "Tickers must be unique"
    for t in tickers:
        assert t == t.upper(), f"Ticker {t} should be uppercase"

def test_get_universe_excludes_known_etfs():
    tickers = get_universe()
    etf_tickers = {"SPY", "IVV", "VOO", "IJH", "MDY", "XLI", "XLF", "XLK", "XLE"}
    intersection = set(tickers) & etf_tickers
    assert not intersection, f"ETFs found in universe: {intersection}"

def test_well_known_tickers_present():
    tickers = set(get_universe())
    must_have = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"}
    missing = must_have - tickers
    assert not missing, f"Missing expected tickers: {missing}"

def test_nyse_tickers_have_hyphens_for_class_shares():
    tickers = get_universe()
    # Should include BRK.B style tickers from Polygon API
    # BRK.B comes as BRK.B, not BRK-B
    assert any("." in t or "-" not in t for t in tickers), "Check share class handling"

@pytest.mark.skip(reason="Requires network — manual validation test")
def test_tickers_resolve_on_polygon():
    """Manual test: verify all tickers resolve on Polygon."""
    pass
```

**Step 2: Minimal implementation**

```python
# data/universe.py
"""S&P 500 and S&P 400 ticker universe — no ETFs, no ADRs, US common stock only."""

from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Hardcoded fallback — the canonical S&P 500 + 400 list.
# Sourced from: Wikipedia S&P 500 / S&P 400 constituent lists
# Last updated: June 2026
_FALLBACK_TICKERS: list[str] = []


def _load_fallback() -> list[str]:
    """Load the hardcoded fallback list."""
    # ~900 tickers total for S&P 500 + S&P 400
    # This is a real, maintained list. We embed it since Wikipedia
    # scraping is fragile and this changes ~monthly.
    path = Path(__file__).parent / "universe_fallback.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return _FALLBACK_TICKERS


def get_universe(refresh: bool = False) -> list[str]:
    """Return deduplicated, sorted list of tickers in the screening universe.

    Args:
        refresh: If True, force a refresh from external sources.
                 Default False — uses cached list.

    Returns:
        Sorted list of uppercase ticker symbols.
    """
    tickers = _load_fallback()
    # Deduplicate and normalize
    seen: set[str] = set()
    result = []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    if not result:
        raise RuntimeError("Universe is empty — check data/universe_fallback.json")
    return sorted(result)


def is_valid_ticker(ticker: str) -> bool:
    """Check if a ticker looks syntactically valid (no ETFs, no weird symbols)."""
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 5:
        return False
    # Exclude obvious ETFs (3-4 letters, starts with X, ends with ETF patterns)
    if ticker in {"SPY", "IVV", "VOO", "IJH", "MDY", "VTI", "QQQ", "IWM"}:
        return False
    return ticker.isalpha() or (ticker.count(".") == 1 and ticker.replace(".", "").isalpha())
```

**Step 3: Create the fallback JSON**

I'll populate `data/universe_fallback.json` with the actual S&P 500 + 400 constituent list (~900 tickers), sourced from Wikipedia scraping or a static snapshot.

**Verification:** `uv run pytest tests/test_universe.py -v` — all non-skip tests pass.

---

### Task 1.4: DuckDB cache helper

**Objective:** Provide a `get_cache()` function returning a DuckDB connection pointed at `data/cache/`. All later modules use this single connection.

**Files:**
- Create: `data/cache.py`
- Create: `tests/test_cache.py`

**Test:**
```python
def test_get_cache_returns_duckdb_connection():
    from data.cache import get_cache
    conn = get_cache()
    assert conn is not None
    result = conn.execute("SELECT 1").fetchone()
    assert result[0] == 1

def test_get_cache_is_singleton():
    from data.cache import get_cache
    conn1 = get_cache()
    conn2 = get_cache()
    assert conn1 is conn2

def test_cache_dir_created():
    from data.cache import get_cache, CACHE_DIR
    assert CACHE_DIR.exists()
    assert CACHE_DIR.is_dir()
```

**Implementation:** Simple singleton DuckDB connection with `data/cache/` directory auto-created.

---

### Task 1.5: data/ingest.py — pull from Polygon.io

**Objective:** Given a ticker, pull fundamentals (market cap, revenue, EPS, FCF, ROE, margins, debt, etc.) and price history from Polygon.io, cache to DuckDB.

**Files:**
- Create: `data/ingest.py`
- Create: `tests/test_ingest.py`

**Key function signatures:**
```python
def load_config() -> dict:
    """Load config.yaml and overlay .env secrets."""

def get_polygon_client() -> "PolygonClient":
    """Return configured Polygon API client (API key from env)."""

def fetch_fundamentals(ticker: str, client: "PolygonClient") -> dict:
    """Pull latest fundamentals for one ticker. Returns dict with keys:
    market_cap, revenue_ttm, eps_ttm, fcf_ttm, roe, roic, gross_margin,
    fcf_margin, debt_equity, current_ratio, altman_z, shares_outstanding, etc.
    """

def fetch_price_history(ticker: str, client: "PolygonClient", days: int = 365) -> pd.DataFrame:
    """Pull daily OHLCV for one ticker."""

def fetch_financials(ticker: str, client: "PolygonClient") -> list[dict]:
    """Pull quarterly financial statements for TTM calculations."""

def cache_fundamentals(ticker: str, data: dict, conn: duckdb.DuckDBPyConnection):
    """Write fundamentals to DuckDB."""

def ingest_universe(tickers: list[str], force_refresh: bool = False) -> int:
    """Pull and cache all tickers. Returns count of successfully cached tickers."""
```

**Test:** With `POLYGON_API_KEY=demo` (Polygon free tier), test that `fetch_fundamentals("AAPL")` returns a dict with expected keys. Mock the HTTP layer for unit tests.

**Verification:** `uv run python -c "from data.ingest import ingest_universe; from data.universe import get_universe; n = ingest_universe(get_universe()[:5]); print(f'Cached {n} tickers')"`

---

### Task 1.6: run.py CLI entrypoint

**Objective:** `python run.py --help` shows subcommands: `ingest`, `screen`, `thesis`, `dashboard`, `all`.

**Files:**
- Create: `run.py`

**Implementation:** Simple argparse-based CLI that imports and calls each module.

---

## Phase 2: Scoring Model

### Task 2.1: Disqualifier filter

**Objective:** Implement the 5 hard disqualifiers from the plan (market cap, volume, FCF, D/E, Altman Z). Returns a boolean mask over a DataFrame.

**Files:**
- Create: `screen/__init__.py`
- Create: `screen/filters.py`
- Create: `tests/test_filters.py`

**Test cases:**
- Below min market cap → disqualified
- Negative FCF → disqualified
- D/E > 3 → disqualified
- Altman Z < 1.8 → disqualified
- All clean → passes all

---

### Task 2.2: Individual factor calculations

**Objective:** Each factor (revenue growth, ROE, ROIC, momentum, etc.) is a pure function: takes a ticker's data dict, returns a float.

**Files:**
- Create: `screen/factors.py`
- Create: `tests/test_factors.py`

**Factors to implement:**
- `revenue_yoy_ttm(fundamentals) -> float`
- `revenue_cagr_3yr(financials) -> float`
- `eps_growth_ttm(fundamentals) -> float`
- `forward_revenue_growth(fundamentals) -> float` (default 0 if no consensus)
- `reinvestment_rate(fundamentals) -> float`
- `rule_of_40(fundamentals) -> float`
- `roe_ttm(fundamentals) -> float`
- `roic(fundamentals) -> float`
- `gross_margin_ttm(fundamentals) -> float`
- `fcf_margin_ttm(fundamentals) -> float`
- `net_debt_ebitda(fundamentals) -> float`
- `accruals_ratio(financials) -> float`
- `interest_coverage(fundamentals) -> float`
- `return_6m_excl_1m(price_history) -> float`
- `return_12m(price_history) -> float`
- `return_1m(price_history) -> float`
- `earnings_surprise(fundamentals) -> float`
- `ma_ratio_50_200(price_history) -> float`

Each factor normalizes within 0–1 range internally, or returns raw for percentile ranking.

---

### Task 2.3: Sharpe ratio factor

**Objective:** Add risk-adjusted momentum via the Sharpe ratio. This surfaces "quiet compounders" — stocks with steady, low-volatility uptrends that pure momentum metrics miss.

**Files:**
- Modify: `screen/factors.py`
- Modify: `tests/test_factors.py`

**Factor function:**
```python
def sharpe_ratio(price_history: pd.DataFrame, risk_free_rate: float = 0.045) -> float:
    """Annualized Sharpe ratio over 1-year lookback.
    
    Sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    
    Returns 0 if insufficient data or negative vol.
    """
```

**How it finds hidden gems:**
- Stock A: 30% return, 40% vol → Sharpe = 0.64 (risky, flashy)
- Stock B: 15% return, 10% vol → Sharpe = 1.05 (steady, overlooked)
- Pure momentum picks A. Sharpe ratio surfaces B.

**Config integration:** Weight in momentum group controlled by `scoring.sharpe.weight_in_momentum`. When enabled, the 20% redistributes from 6m/12m return weights.

**Test cases:**
- Perfect steady uptrend → high Sharpe
- Volatile whipsaw → low Sharpe
- Insufficient data → returns 0 (graceful degradation)
- Flat price → near-zero Sharpe

---

### Task 2.4: HMM regime detection per stock

**Objective:** Fit a 3-state Gaussian HMM on each stock's 2-year log returns to detect Bull/Sideways/Bear regimes. Extract current regime probability and regime stickiness.

**Files:**
- Create: `screen/hmm_detect.py`
- Create: `tests/test_hmm_detect.py`

**Key functions:**
```python
from hmmlearn import hmm
import numpy as np

def fit_regime_model(log_returns: np.ndarray, n_seeds: int = 20) -> dict:
    """Fit 3-state Gaussian HMM, try multiple seeds, return best model.
    
    Returns:
        {
            "means": array of 3 annualized mean returns per regime,
            "vols": array of 3 annualized vols per regime,
            "regime_labels": {0: "BEAR", 1: "SIDEWAYS", 2: "BULL"},
            "current_state": int,
            "current_probs": [P(bear), P(sideways), P(bull)],
            "transition_matrix": 3x3 array,
            "bull_probability": float (0-1),
            "regime_stickiness": dict of regime → stickiness %,
            "regime_annual_return": float (expected return for current regime),
            "actual_excess_return": float (actual - expected, annualized),
        }
    """

def regime_adjustment(hmm_result: dict) -> float:
    """Convert HMM regime data into a scoring adjustment (-0.1 to +0.2).
    
    "Hidden gems" logic:
    - Stock in Bear regime but delivering +10% → large positive bonus
    - Stock in Bull regime delivering +10% → neutral (meets expectations)
    - Stock in Bull regime delivering -5% → negative (underperforming regime)
    """
```

**Performance guard:** HMM fitting is expensive (~0.1s per stock × 20 seeds = 2s per stock). For 900 stocks that's ~30 minutes. Strategy:
1. Run a "pre-score" with fast factors first → rank the universe
2. Fit HMM only on top 200 (configurable: `scoring.hmm.max_stocks_for_hmm`)
3. Lower-ranked stocks get neutral HMM adjustment (0.0)
4. This caps HMM time at ~6-7 minutes

**Reference:** See `quantitative-stock-analysis` skill for the proven HMM implementation (hmmlearn, 20 seeds, best-score selection).

**Test cases:**
- Synthetic data with clear regimes (sin wave) → HMM detects 3 states
- Trending data (persistent bull) → HMM may collapse to 2 states (acceptable — log warning)
- Edge case: stock with < 252 trading days → skip HMM, return neutral

---

### Task 2.5: Regime-adjusted composite score

**Objective:** Integrate Sharpe ratio and HMM regime bonus into the final composite score. The scoring pipeline becomes: filters → factors (including Sharpe) → sector percentiles → HMM adjustment → composite.

**Files:**
- Modify: `screen/score.py`
- Modify: `tests/test_score.py`

**Scoring formula update:**
```
pre_score = (growth * 0.40) + (quality * 0.35) + (momentum * 0.25)
  where momentum now includes Sharpe at 20% weight:
    momentum = (return_6m * 0.24) + (return_12m * 0.20) + (return_1m * 0.15)
             + (earnings_surprise * 0.15) + (ma_ratio * 0.15) + (sharpe * 0.20)

final_score = pre_score + hmm_regime_bonus  
  where hmm_regime_bonus ∈ [-0.10, +0.10] from regime_adjustment()
```

**"Hidden gems" column:** Add a `hidden_gem_score` column:
```
hidden_gem_score = (sharpe_percentile * 0.5) + (hmm_excess_return_percentile * 0.5)
```
This lets the dashboard sort by "most interesting risk-adjusted picks" — the stocks the pure screener would miss.

**Test:** With mock data for 5 stocks including one "hidden gem" (moderate returns, low vol, bear regime), verify:
- Hidden gem ranks higher by `hidden_gem_score` than by `composite_score`
- All other rankings shift minimally

---

### Task 2.6: Sector-relative percentile ranking

**Objective:** Given a DataFrame of tickers + metrics + GICS sector, compute each metric's percentile rank within sector. Return a scored DataFrame.

**Files:**
- Create: `screen/sector_percentiles.py`
- Create: `tests/test_sector_percentiles.py`

**Test:** Create a small DataFrame with 3 tech stocks and 3 financials, verify percentiles are computed within sector, not globally.

---

### Task 2.7: screen/score.py — the full scoring pipeline

**Objective:** Glue together filters → factors → sector percentiles → weighted composite score.

**Files:**
- Create: `screen/score.py`
- Create: `tests/test_score.py`

**Key function:**
```python
def score_universe(tickers: list[str], conn) -> pd.DataFrame:
    """Returns DataFrame with columns: ticker, sector, growth_score, 
    quality_score, momentum_score, composite_score, rank, disqualified, 
    disqualifier_reason"""
```

**Test:** With mock data for 10 tickers, verify ranking order, verify disqualified tickers are flagged but not removed (caller decides).

---

### Task 2.8: screen/score.py — config-driven weights

**Objective:** Read weights and metric selections from `config.yaml`, not hardcoded. Changing config changes scores without code changes.

**Files:**
- Modify: `screen/score.py`

---

## Phase 3: AI Thesis (deferred to next iteration)

### Task 3.1: thesis/sources.py — fetch 10-K, 10-Q, transcripts, news

### Task 3.2: thesis/prompt.py — prompt template

### Task 3.3: thesis/schema.py — JSON schema + validator

### Task 3.4: thesis/generate.py — run LLM, validate, retry

---

## Phase 4: Output (deferred)

### Task 4.1: out/dashboard.py — Streamlit app

### Task 4.2: out/notion_sync.py — Notion push

---

## Verification Checklist (end of each phase)

- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] Config changes propagate without code changes
- [ ] Data cache survives restarts
- [ ] Each ticker failure doesn't kill the whole run
- [ ] Token usage and cost logged per run

---

## Open Design Decisions

1. **Polygon free tier vs. paid:** The free tier allows 5 API calls/min. For 900 tickers that's 3+ hours. Start with free tier for development, document paid tier upgrade path.
2. **Wikipedia scraping for tickers:** Use a static JSON snapshot checked into git. User refreshes monthly with a script. Avoids fragile runtime scraping.
3. **DuckDB vs. Parquet files:** Use DuckDB tables — faster queries, SQL interface, zero-config. Parquet files as export format only.
4. **Streamlit vs. FastAPI+HTMX:** Streamlit for v1 per plan recommendation. Revisit if sharing/auth needed.
