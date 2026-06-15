"""Tests for screen/filters.py — hard disqualifiers."""
import pandas as pd
import pytest
from screen.filters import apply_disqualifiers


def _make_row(**overrides) -> dict:
    """Factory for a passing row — override specific fields to trigger disqualifiers."""
    base = {
        "ticker": "TEST",
        "market_cap": 10e9,       # $10B — passes
        "avg_volume_20d": 10e6,   # 10M — passes
        "fcf_ttm": 1e9,           # positive — passes
        "debt_equity": 1.0,       # low — passes
        "altman_z": 2.5,          # above 1.8 — passes
    }
    base.update(overrides)
    return base


def test_all_pass_returns_no_disqualifications():
    df = pd.DataFrame([_make_row()])
    result = apply_disqualifiers(df)
    assert result["disqualified"].iloc[0] == False
    assert result["disqualifier_reason"].iloc[0] == ""


def test_market_cap_below_threshold_disqualifies():
    df = pd.DataFrame([_make_row(market_cap=1.5e9)])  # $1.5B
    result = apply_disqualifiers(df)
    assert result["disqualified"].iloc[0] == True
    assert "market cap" in result["disqualifier_reason"].iloc[0].lower()


def test_negative_fcf_disqualifies():
    df = pd.DataFrame([_make_row(fcf_ttm=-100e6)])  # -$100M FCF
    result = apply_disqualifiers(df)
    assert result["disqualified"].iloc[0] == True
    assert "free cash flow" in result["disqualifier_reason"].iloc[0].lower()


def test_high_debt_equity_disqualifies():
    df = pd.DataFrame([_make_row(debt_equity=4.0)])  # D/E of 4
    result = apply_disqualifiers(df)
    assert result["disqualified"].iloc[0] == True
    assert "debt" in result["disqualifier_reason"].iloc[0].lower()


def test_low_altman_z_disqualifies():
    df = pd.DataFrame([_make_row(altman_z=1.2)])  # Z < 1.8
    result = apply_disqualifiers(df)
    assert result["disqualified"].iloc[0] == True
    assert "altman" in result["disqualifier_reason"].iloc[0].lower()


def test_low_volume_disqualifies():
    df = pd.DataFrame([_make_row(avg_volume_20d=2e6)])  # $2M volume
    result = apply_disqualifiers(df)
    assert result["disqualified"].iloc[0] == True
    assert "volume" in result["disqualifier_reason"].iloc[0].lower()


def test_multiple_failures_reports_first_only():
    """If multiple disqualifiers trigger, report the first one found."""
    df = pd.DataFrame([_make_row(market_cap=0.5e9, fcf_ttm=-1e6, debt_equity=5.0)])
    result = apply_disqualifiers(df)
    assert result["disqualified"].iloc[0] == True
    # Should report the first failure (market cap)
    assert "market cap" in result["disqualifier_reason"].iloc[0].lower()


def test_mixed_universe_preserves_passing_tickers():
    df = pd.DataFrame([
        _make_row(ticker="GOOD"),
        _make_row(ticker="BAD", fcf_ttm=-1e6),
        _make_row(ticker="ALSO_GOOD"),
    ])
    result = apply_disqualifiers(df)
    assert result["disqualified"].tolist() == [False, True, False]


def test_missing_field_defaults_to_pass():
    """If a field is missing (NaN), don't disqualify — the scoring step handles it."""
    df = pd.DataFrame([_make_row()])
    del df["altman_z"]  # remove the column
    result = apply_disqualifiers(df)
    # Should not crash, and should not disqualify for missing data
    assert result["disqualified"].iloc[0] == False


def test_configurable_thresholds():
    """Thresholds can be overridden via kwargs."""
    df = pd.DataFrame([_make_row(market_cap=3e9)])  # $3B — passes default $2B
    # Tighten to $5B
    result = apply_disqualifiers(df, min_market_cap_b=5.0)
    assert result["disqualified"].iloc[0] == True
    assert "market cap" in result["disqualifier_reason"].iloc[0].lower()
