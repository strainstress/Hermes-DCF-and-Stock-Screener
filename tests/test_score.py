"""Tests for screen/score.py — full scoring pipeline."""
import pandas as pd
import numpy as np
import pytest
from screen.score import score_universe


def _make_test_df(n: int = 10) -> pd.DataFrame:
    """Create a test DataFrame with required columns."""
    np.random.seed(42)
    tickers = [f"TICK{i}" for i in range(n)]
    sectors = ["Tech"] * (n // 2) + ["Finance"] * (n - n // 2)

    return pd.DataFrame({
        "ticker": tickers,
        "sector": sectors,
        "market_cap": np.random.uniform(2e9, 500e9, n),
        "avg_volume_20d": np.random.uniform(5e6, 50e6, n),
        "fcf_ttm": np.random.uniform(100e6, 10e9, n),
        "debt_equity": np.random.uniform(0.2, 2.5, n),
        "altman_z": np.random.uniform(2.0, 5.0, n),
        "revenue_ttm": np.random.uniform(1e9, 100e9, n),
        "revenue_yoy_ttm": np.random.uniform(-0.1, 0.3, n),
        "eps_yoy_ttm": np.random.uniform(-0.2, 0.4, n),
        "forward_revenue_growth": np.random.uniform(0.0, 0.2, n),
        "capex_ttm": np.random.uniform(100e6, 5e9, n),
        "rd_expense_ttm": np.random.uniform(0, 3e9, n),
        "fcf_margin": np.random.uniform(0.05, 0.25, n),
        "roe": np.random.uniform(0.05, 0.3, n),
        "roic": np.random.uniform(0.05, 0.25, n),
        "gross_margin": np.random.uniform(0.3, 0.7, n),
        "total_debt": np.random.uniform(1e9, 20e9, n),
        "cash_and_equivalents": np.random.uniform(500e6, 10e9, n),
        "ebitda_ttm": np.random.uniform(500e6, 20e9, n),
        "interest_coverage": np.random.uniform(3, 30, n),
        "earnings_surprise_avg": np.random.uniform(-0.05, 0.1, n),
    })


class TestScoreUniverse:
    def test_returns_dataframe_with_expected_columns(self):
        df = _make_test_df(5)
        result = score_universe(df)
        expected_cols = {
            "ticker", "composite_score", "growth_score",
            "quality_score", "momentum_score", "rank",
            "disqualified", "disqualifier_reason",
        }
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_scores_in_reasonable_range(self):
        df = _make_test_df(10)
        result = score_universe(df)
        passing = result[~result["disqualified"]]
        if len(passing) > 0:
            # Scores should be roughly 0-100 after percentile ranking
            assert result["composite_score"].max() >= 0
            assert result["composite_score"].max() <= 110, f"Max score too high: {result['composite_score'].max()}"

    def test_disqualified_marked(self):
        df = _make_test_df(5)
        df.loc[0, "fcf_ttm"] = -1e9  # disqualify one
        result = score_universe(df)
        assert result["disqualified"].iloc[0] or any(result["disqualified"]), (
            "At least one ticker should be disqualified with negative FCF"
        )

    def test_ranking_is_monotonic(self):
        df = _make_test_df(10)
        result = score_universe(df)
        passing = result[~result["disqualified"]].sort_values("rank")
        scores = passing["composite_score"].values
        # Scores should be non-increasing as rank increases
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Rank {i} score {scores[i]} < rank {i+1} score {scores[i+1]}"
            )

    def test_hidden_gem_score_column_exists(self):
        df = _make_test_df(5)
        result = score_universe(df)
        assert "hidden_gem_score" in result.columns

    def test_empty_dataframe(self):
        df = _make_test_df(0)
        result = score_universe(df)
        assert len(result) == 0

    def test_all_disqualified_handled(self):
        df = _make_test_df(3)
        df["fcf_ttm"] = -1e9  # all negative FCF
        result = score_universe(df)
        assert result["disqualified"].sum() == 3
