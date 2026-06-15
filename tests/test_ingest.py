"""Tests for data/ingest.py — Polygon.io data pipeline."""
import json
from unittest.mock import patch, MagicMock
import pytest
from data.ingest import (
    load_config,
    PolygonClient,
    fetch_fundamentals,
    cache_fundamentals,
    ingest_universe,
)


# ── Sample Polygon API responses ───────────────────────────────

SAMPLE_TICKER_DETAILS = {
    "results": {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "market_cap": 3_000_000_000_000,
        "sic_description": "Electronic Computers",
        "primary_exchange": "NASDAQ",
        "currency_name": "usd",
        "sic_code": "3571",
    }
}

SAMPLE_FINANCIALS = {
    "results": [
        {
            "filing_date": "2025-12-31",
            "fiscal_year": "2025",
            "fiscal_period": "FY",
            "financials": {
                "income_statement": {
                    "revenues": {"value": 400_000_000_000},
                    "cost_of_revenue": {"value": 220_000_000_000},
                    "gross_profit": {"value": 180_000_000_000},
                    "operating_expenses": {"value": 50_000_000_000},
                    "operating_income_loss": {"value": 130_000_000_000},
                    "net_income_loss": {"value": 100_000_000_000},
                    "diluted_earnings_per_share": {"value": 6.50},
                    "income_tax_expense_benefit": {"value": 20_000_000_000},
                    "interest_expense_operating": {"value": 4_000_000_000},
                },
                "balance_sheet": {
                    "assets": {"value": 400_000_000_000},
                    "current_assets": {"value": 150_000_000_000},
                    "liabilities": {"value": 300_000_000_000},
                    "current_liabilities": {"value": 120_000_000_000},
                    "equity": {"value": 100_000_000_000},
                    "long_term_debt": {"value": 95_000_000_000},
                },
                "cash_flow_statement": {
                    "net_cash_flow_from_operating_activities": {"value": 120_000_000_000},
                    "net_cash_flow_from_investing_activities": {"value": -30_000_000_000},
                    "net_cash_flow_from_financing_activities": {"value": -80_000_000_000},
                },
            },
        }
    ]
}

SAMPLE_AGGS = {
    "results": [
        {
            "ticker": "AAPL",
            "volume": 50_000_000.0,
            "open": 185.0,
            "close": 190.0,
            "high": 191.0,
            "low": 184.0,
            "timestamp": 1_717_000_000_000_000,  # nanos
            "transactions": 500_000,
        }
    ]
}


# ── Tests ──────────────────────────────────────────────────────


class TestLoadConfig:
    def test_loads_yaml_config(self, tmp_path):
        """Load config.yaml and merge .env secrets."""
        # Not a unit test — just verifies the function exists
        cfg = load_config()
        assert "universe" in cfg
        assert "scoring" in cfg


class TestPolygonClient:
    def test_client_stores_api_key(self):
        client = PolygonClient("test_key_123")
        assert client.api_key == "test_key_123"

    def test_get_headers(self):
        client = PolygonClient("key_abc")
        headers = client._headers()
        assert "Authorization" in headers
        assert "key_abc" in headers["Authorization"]


class TestFetchFundamentals:
    @patch("data.ingest.requests.get")
    def test_returns_structured_dict(self, mock_get):
        """fetch_fundamentals returns expected keys."""
        mock_ticker = MagicMock()
        mock_ticker.status_code = 200
        mock_ticker.json.return_value = SAMPLE_TICKER_DETAILS

        mock_fin = MagicMock()
        mock_fin.status_code = 200
        mock_fin.json.return_value = SAMPLE_FINANCIALS

        mock_get.side_effect = [mock_ticker, mock_fin]

        client = PolygonClient("test_key")
        result = fetch_fundamentals("AAPL", client)

        # Must have all the keys the scoring model expects
        assert "ticker" in result
        assert result["ticker"] == "AAPL"
        assert "market_cap" in result
        assert "revenue_ttm" in result
        assert "eps_ttm" in result
        assert "gross_margin" in result
        assert "fcf_ttm" in result

    @patch("data.ingest.requests.get")
    def test_handles_api_error_gracefully(self, mock_get):
        """API errors should return None, not crash."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limit exceeded"
        mock_resp.raise_for_status.side_effect = Exception("429")
        mock_get.return_value = mock_resp

        client = PolygonClient("test_key")
        result = fetch_fundamentals("BAD", client)
        assert result is None


class TestCacheFundamentals:
    def test_writes_to_duckdb(self):
        """cache_fundamentals writes data that can be read back."""
        from data.cache import get_cache, reset_cache

        reset_cache()
        conn = get_cache()

        data = {
            "ticker": "AAPL",
            "market_cap": 3_000_000_000_000,
            "revenue_ttm": 400_000_000_000,
            "gross_margin": 0.45,
            "fcf_ttm": 90_000_000_000,
        }
        cache_fundamentals("AAPL", data, conn)

        rows = conn.execute(
            "SELECT * FROM fundamentals WHERE ticker = 'AAPL'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "AAPL"  # ticker is first column
        # market_cap is third column (0=ticker, 1=name, 2=market_cap)
        assert rows[0][2] == 3_000_000_000_000


class TestIngestUniverse:
    @patch("data.ingest.time.sleep")
    @patch("data.ingest.get_polygon_client")
    @patch("data.ingest.fetch_fundamentals")
    def test_reports_cached_count(self, mock_fetch, mock_client, mock_sleep):
        """ingest_universe returns count of successfully cached tickers."""
        mock_client.return_value = MagicMock()
        mock_fetch.return_value = {
            "ticker": "TEST",
            "market_cap": 10e9,
            "revenue_ttm": 5e9,
            "gross_margin": 0.40,
            "fcf_ttm": 1e9,
        }
        from data.cache import reset_cache

        reset_cache()
        count = ingest_universe(["TEST1", "TEST2", "TEST3"], force_refresh=True)
        assert count == 3
