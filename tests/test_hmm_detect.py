"""Tests for screen/hmm_detect.py — HMM regime detection."""
import numpy as np
import pytest
from screen.hmm_detect import fit_regime_model, regime_adjustment


def _generate_regime_data(
    n_bear: int = 100,
    n_sideways: int = 100,
    n_bull: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic log returns with clear regime shifts."""
    np.random.seed(seed)
    bear = np.random.normal(-0.002, 0.02, n_bear)   # -50% annual, 32% vol
    sideways = np.random.normal(0.0002, 0.01, n_sideways)  # 5% annual, 16% vol
    bull = np.random.normal(0.0015, 0.012, n_bull)   # 38% annual, 19% vol
    return np.concatenate([bear, sideways, bull])


def test_fit_regime_model_returns_dict():
    """fit_regime_model should return a structured dict."""
    returns = _generate_regime_data()
    result = fit_regime_model(returns, n_seeds=5)
    assert isinstance(result, dict)
    assert "means" in result
    assert "vols" in result
    assert "current_state" in result
    assert "current_probs" in result
    assert "bull_probability" in result
    assert "transition_matrix" in result


def test_detects_three_regimes():
    """With clear synthetic data, HMM should find ~3 regimes."""
    returns = _generate_regime_data(n_bear=200, n_sideways=200, n_bull=200)
    result = fit_regime_model(returns, n_seeds=10)
    # Should have 3 components
    assert len(result["means"]) == 3
    # Means should be ordered Bear < Sideways < Bull
    assert result["means"][0] < result["means"][1] < result["means"][2]


def test_bull_regime_has_highest_mean():
    """The Bull regime should have the highest mean of the three."""
    returns = _generate_regime_data(n_bear=300, n_sideways=300, n_bull=300)
    result = fit_regime_model(returns, n_seeds=10)
    # Bull regime (index 2) should have the highest mean
    assert result["means"][2] > result["means"][0], "Bull mean should exceed Bear mean"


def test_insufficient_data_returns_neutral():
    """Less than ~50 data points should return a neutral result."""
    returns = np.random.normal(0.001, 0.01, 20)
    result = fit_regime_model(returns, n_seeds=3)
    # Should not crash, should return something usable
    assert result["bull_probability"] >= 0
    assert result["bull_probability"] <= 1


def test_single_regime_trending_data():
    """Persistent uptrend may collapse to 2 effective regimes — should handle gracefully."""
    np.random.seed(123)
    # Strong persistent uptrend
    returns = np.random.normal(0.002, 0.01, 500)  # Only one real regime
    result = fit_regime_model(returns, n_seeds=5)
    # Should not crash
    assert 1 <= len(set(result.get("regime_labels", {}).values())) <= 3


# ── regime_adjustment ──────────────────────────────────────────


def test_regime_adjustment_bonus_for_beating_bear():
    """Stock in Bear regime but returning +10% gets positive bonus."""
    hmm_result = {
        "bull_probability": 0.05,  # very bearish
        "regime_annual_return": -0.30,  # -30% expected in bear
        "actual_excess_return": 0.40,  # beating by 40% → +10% actual vs -30% expected
    }
    adj = regime_adjustment(hmm_result)
    assert adj > 0, f"Expected positive bonus, got {adj}"


def test_regime_adjustment_neutral_for_meeting_bull():
    """Stock in Bull regime meeting expectations gets near-zero adjustment."""
    hmm_result = {
        "bull_probability": 0.95,
        "regime_annual_return": 0.25,
        "actual_excess_return": 0.02,  # barely beating
    }
    adj = regime_adjustment(hmm_result)
    assert -0.05 <= adj <= 0.05, f"Expected near-zero, got {adj}"


def test_regime_adjustment_penalty_for_underperforming():
    """Stock in Bull regime but delivering -5% gets negative adjustment."""
    hmm_result = {
        "bull_probability": 0.90,
        "regime_annual_return": 0.30,
        "actual_excess_return": -0.35,  # -5% actual vs 30% expected
    }
    adj = regime_adjustment(hmm_result)
    assert adj < 0, f"Expected negative penalty, got {adj}"


def test_regime_adjustment_bounded():
    """Adjustment should be within [-0.15, +0.15]."""
    for excess in [-2.0, -0.5, 0, 0.5, 2.0]:
        hmm_result = {
            "bull_probability": 0.5,
            "regime_annual_return": 0.10,
            "actual_excess_return": excess,
        }
        adj = regime_adjustment(hmm_result)
        assert -0.15 <= adj <= 0.15, f"excess={excess}, adj={adj} out of bounds"
