"""Tests for screen/factors.py — individual factor calculations."""
import numpy as np
import pytest
from screen.factors import (
    revenue_yoy_ttm,
    revenue_cagr_3yr,
    eps_growth_ttm,
    reinvestment_rate,
    rule_of_40,
    roe_ttm,
    roic,
    gross_margin_ttm,
    fcf_margin_ttm,
    net_debt_ebitda,
    accruals_ratio,
    interest_coverage,
    return_6m_excl_1m,
    return_12m,
    return_1m,
    earnings_surprise,
    ma_ratio_50_200,
    sharpe_ratio,
    compute_all_factors,
)


# ── Helpers ────────────────────────────────────────────────────


def _make_fundamentals(**overrides) -> dict:
    base = {
        "revenue_yoy_ttm": 0.15,
        "eps_yoy_ttm": 0.20,
        "forward_revenue_growth": 0.12,
        "revenue_ttm": 100e9,
        "capex_ttm": 5e9,
        "rd_expense_ttm": 3e9,
        "fcf_margin": 0.18,
        "roe": 0.25,
        "roic": 0.20,
        "gross_margin": 0.55,
        "total_debt": 30e9,
        "cash_and_equivalents": 10e9,
        "ebitda_ttm": 25e9,
        "interest_coverage": 15.0,
        "earnings_surprise_avg": 0.03,
    }
    base.update(overrides)
    return base


def _make_price_bars(prices: list[float]) -> list[dict]:
    """Create price bars sorted by date."""
    return [
        {"date": 1_000_000_000_000 + i * 86_400_000_000_000, "close": p}
        for i, p in enumerate(prices)
    ]


def _make_financials(quarterly_revs: list[float]) -> list[dict]:
    """Create mock quarterly financials with given revenue values."""
    return [
        {
            "financials": {
                "income_statement": {
                    "revenues": {"value": rev},
                    "net_income_loss": {"value": rev * 0.1},
                },
                "cash_flow_statement": {
                    "net_cash_flow_from_operating_activities": {"value": rev * 0.12}
                },
                "balance_sheet": {"assets": {"value": rev * 2}},
            }
        }
        for rev in quarterly_revs
    ]


# ── Growth Factor Tests ────────────────────────────────────────


class TestGrowthFactors:
    def test_revenue_yoy_ttm_returns_value(self):
        assert revenue_yoy_ttm({"revenue_yoy_ttm": 0.15}) == 0.15

    def test_revenue_yoy_ttm_defaults_to_zero(self):
        assert revenue_yoy_ttm({}) == 0.0

    def test_revenue_cagr_3yr_calculates_cagr(self):
        # Revenue doubled over 3 years → CAGR = 2^(1/3) - 1 ≈ 0.2599
        revs = [200, 150, 120, 100]  # 4 quarters: latest → oldest, need 12 min
        revs_12 = revs * 3  # 12 quarters
        fin = _make_financials(revs_12)
        cagr = revenue_cagr_3yr(fin)
        assert cagr > 0.2

    def test_revenue_cagr_insufficient_data(self):
        assert revenue_cagr_3yr([]) == 0.0

    def test_reinvestment_rate_computes(self):
        f = _make_fundamentals(revenue_ttm=100e9, capex_ttm=5e9, rd_expense_ttm=3e9)
        rate = reinvestment_rate(f)
        assert rate == pytest.approx(8e9 / 100e9)  # 0.08

    def test_rule_of_40(self):
        f = _make_fundamentals(revenue_yoy_ttm=0.25, fcf_margin=0.18)
        r40 = rule_of_40(f)
        assert r40 == pytest.approx(43.0)  # 25 + 18 = 43


# ── Quality Factor Tests ───────────────────────────────────────


class TestQualityFactors:
    def test_net_debt_ebitda_low_is_good(self):
        # Low leverage → low net_debt/ebitda
        f = _make_fundamentals(total_debt=10e9, cash_and_equivalents=8e9, ebitda_ttm=20e9)
        ratio = net_debt_ebitda(f)
        assert ratio == pytest.approx(0.1)  # (10-8)/20 = 0.1

    def test_net_debt_ebitda_negative_ebitda(self):
        f = _make_fundamentals(ebitda_ttm=0)
        ratio = net_debt_ebitda(f)
        assert ratio == 999.0  # effectively infinite

    def test_accruals_ratio_zero_for_clean_earnings(self):
        # Net income = OpCF → accruals = 0
        fin = [{
            "financials": {
                "income_statement": {"net_income_loss": {"value": 100}},
                "cash_flow_statement": {"net_cash_flow_from_operating_activities": {"value": 100}},
                "balance_sheet": {"assets": {"value": 1000}},
            }
        }]
        assert accruals_ratio(fin) == 0.0

    def test_roe_returns_value(self):
        assert roe_ttm({"roe": 0.25}) == 0.25

    def test_interest_coverage(self):
        assert interest_coverage({"interest_coverage": 12.0}) == 12.0


# ── Momentum Factor Tests ──────────────────────────────────────


class TestMomentumFactors:
    def test_return_12m_positive(self):
        # Price went from $100 → $120 over 252 days
        prices = [100.0] * 252 + [120.0]  # len = 253
        # Actually we need exactly 252 bars minimum
        bars = _make_price_bars([100.0] + [100.0] * 250 + [120.0])  # start=100, end=120
        ret = return_12m(bars)
        assert ret == pytest.approx(0.20)

    def test_return_6m_excl_1m(self):
        # 6 months ago: $100, 1 month ago: $115, today: $118
        # 126 bars total: bars[-126]=$100, bars[-21]=$115 → return = 15%
        prices = [100.0] * (252 - 126) + [100.0] + [100.0] * (126 - 21 - 1) + [115.0] + [115.0] * 20
        bars = _make_price_bars(prices)
        ret = return_6m_excl_1m(bars)
        assert ret > 0.10

    def test_ma_ratio_bullish(self):
        # 50-day MA > 200-day MA
        prices = [90.0] * 150 + [100.0] * 50  # 200 days
        # MA200 = (90*150 + 100*50)/200 = 92.5, MA50 = 100
        bars = _make_price_bars(prices)
        ratio = ma_ratio_50_200(bars)
        assert ratio > 1.0

    def test_ma_ratio_insufficient_data(self):
        bars = _make_price_bars([100.0] * 50)
        assert ma_ratio_50_200(bars) == 1.0  # neutral

    def test_earnings_surprise(self):
        assert earnings_surprise({"earnings_surprise_avg": 0.05}) == 0.05


# ── Sharpe Ratio Tests ─────────────────────────────────────────


class TestSharpeRatio:
    def test_steady_uptrend_high_sharpe(self):
        """Steady 0.06% daily return with 1% vol → high Sharpe."""
        np.random.seed(42)
        daily_rets = np.random.normal(0.0006, 0.01, 252)
        prices = 100 * np.cumprod(1 + daily_rets)
        bars = _make_price_bars(prices.tolist())
        sr = sharpe_ratio(bars, risk_free_rate=0.045)
        # Should be positive (low vol, consistent returns)
        assert sr > 0

    def test_volatile_whipsaw_low_sharpe(self):
        """High volatility, near-zero drift → low Sharpe."""
        np.random.seed(99)
        daily_rets = np.random.normal(0.0001, 0.04, 252)  # 4% daily vol!
        prices = 100 * np.cumprod(1 + daily_rets)
        bars = _make_price_bars(prices.tolist())
        sr_volatile = sharpe_ratio(bars, risk_free_rate=0.045)

        # Compare with steady uptrend
        np.random.seed(42)
        steady = np.random.normal(0.0006, 0.01, 252)
        steady_prices = 100 * np.cumprod(1 + steady)
        steady_bars = _make_price_bars(steady_prices.tolist())
        sr_steady = sharpe_ratio(steady_bars, risk_free_rate=0.045)

        # The steady stock should have higher Sharpe than the volatile one
        assert sr_steady > sr_volatile, (
            f"Expected Sharpe(steady)={sr_steady:.3f} > Sharpe(volatile)={sr_volatile:.3f}"
        )

    def test_insufficient_data_returns_zero(self):
        bars = _make_price_bars([100.0] * 10)
        assert sharpe_ratio(bars) == 0.0

    def test_flat_price_near_zero(self):
        """Flat price → near-zero Sharpe."""
        prices = [100.0] * 300
        bars = _make_price_bars(prices)
        sr = sharpe_ratio(bars, risk_free_rate=0.045)
        # Should be negative or near-zero (no return, tiny vol)
        assert sr < 0.1


# ── compute_all_factors ────────────────────────────────────────


class TestComputeAllFactors:
    def test_returns_all_expected_keys(self):
        result = compute_all_factors(_make_fundamentals())
        expected_keys = {
            "revenue_yoy_ttm", "revenue_cagr_3yr", "eps_growth_ttm",
            "forward_revenue_growth", "reinvestment_rate", "rule_of_40",
            "roe_ttm", "roic", "gross_margin_ttm", "fcf_margin_ttm",
            "net_debt_ebitda", "accruals_ratio", "interest_coverage",
            "return_6m_excl_1m", "return_12m", "return_1m",
            "earnings_surprise", "ma_ratio_50_200", "sharpe_ratio",
        }
        assert set(result.keys()) == expected_keys

    def test_all_values_are_floats(self):
        result = compute_all_factors(_make_fundamentals())
        for k, v in result.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"
