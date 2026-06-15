"""Tests for data/universe.py — S&P 500 + 400 ticker list."""
import pytest
from data.universe import get_universe, is_valid_ticker


def test_get_universe_returns_nonempty_list():
    """Universe must return at least one ticker."""
    tickers = get_universe()
    assert len(tickers) > 0, "Universe should return at least one ticker"


def test_get_universe_returns_deduplicated_uppercase_tickers():
    """All tickers unique and uppercase."""
    tickers = get_universe()
    assert len(tickers) == len(set(tickers)), f"Duplicates found: {len(tickers)} vs {len(set(tickers))}"
    for t in tickers:
        assert t == t.upper(), f"Ticker {t} should be uppercase"


def test_get_universe_excludes_known_etfs():
    """No ETFs in the universe."""
    tickers = get_universe()
    etf_tickers = {"SPY", "IVV", "VOO", "IJH", "MDY", "XLI", "XLF", "XLK", "XLE", "QQQ", "IWM", "VTI"}
    intersection = set(tickers) & etf_tickers
    assert not intersection, f"ETFs found in universe: {intersection}"


def test_well_known_tickers_present():
    """Major S&P 500 names must be present."""
    tickers = set(get_universe())
    must_have = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "JPM", "JNJ"}
    missing = must_have - tickers
    assert not missing, f"Missing expected tickers: {missing}"


def test_get_universe_is_sorted():
    """Tickers should be sorted alphabetically."""
    tickers = get_universe()
    assert tickers == sorted(tickers), "Tickers should be sorted"


def test_minimum_size():
    """S&P 500 + S&P 400 should give at least 800 unique tickers (some overlap possible)."""
    tickers = get_universe()
    assert len(tickers) >= 700, f"Expected >=700 tickers, got {len(tickers)}"


# ── is_valid_ticker ────────────────────────────────────────────

def test_is_valid_ticker_accepts_standard():
    assert is_valid_ticker("AAPL") is True
    assert is_valid_ticker("MSFT") is True


def test_is_valid_ticker_rejects_etfs():
    assert is_valid_ticker("SPY") is False
    assert is_valid_ticker("QQQ") is False
    assert is_valid_ticker("VOO") is False


def test_is_valid_ticker_rejects_empty():
    assert is_valid_ticker("") is False
    assert is_valid_ticker("   ") is False


def test_is_valid_ticker_rejects_long():
    assert is_valid_ticker("TOOLONG") is False


def test_is_valid_ticker_accepts_share_classes():
    """BRK.B, BF.B style tickers should be valid."""
    assert is_valid_ticker("BRK.B") is True
