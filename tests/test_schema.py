"""Tests for thesis/schema.py — analyst report schema validation."""
import pytest
from thesis.schema import validate_report, AnalystReport, QualitativeScores


class TestValidateReport:
    def test_valid_report_passes(self):
        data = {
            "tldr": "Strong company with growing moat.",
            "bull_case": ["Revenue growth accelerating", "Market share gains"],
            "bear_case": ["Valuation stretched", "Regulatory risk"],
            "key_metrics_to_watch": ["Revenue growth", "FCF margin"],
            "qualitative_scores": {
                "moat": 8,
                "management": 7,
                "capital_allocation": 6,
                "industry_position": 8,
            },
            "peer_comparison": "Outperforms peers on growth, trails on margins.",
            "verdict": "Buy",
            "confidence": 75,
            "sources_cited": ["10-K p.42", "Q2 transcript"],
        }
        valid, report, err = validate_report(data)
        assert valid, f"Expected valid, got: {err}"
        assert report.tldr == "Strong company with growing moat."
        assert report.verdict == "Buy"
        assert report.qualitative_scores.moat == 8

    def test_missing_field_fails(self):
        data = {"tldr": "Missing everything else"}
        valid, report, err = validate_report(data)
        assert not valid

    def test_invalid_verdict_fails(self):
        data = {
            "tldr": "test",
            "bull_case": ["a"],
            "bear_case": ["b"],
            "key_metrics_to_watch": ["c"],
            "qualitative_scores": {"moat": 5, "management": 5, "capital_allocation": 5, "industry_position": 5},
            "peer_comparison": "test",
            "verdict": "Amazing",  # not in allowed values
            "confidence": 50,
            "sources_cited": [],
        }
        valid, _, _ = validate_report(data)
        assert not valid

    def test_score_out_of_range_fails(self):
        data = {
            "tldr": "test",
            "bull_case": ["a"],
            "bear_case": ["b"],
            "key_metrics_to_watch": ["c"],
            "qualitative_scores": {"moat": 15, "management": 5, "capital_allocation": 5, "industry_position": 5},
            "peer_comparison": "test",
            "verdict": "Watch",
            "confidence": 50,
            "sources_cited": [],
        }
        valid, _, _ = validate_report(data)
        assert not valid

    def test_json_string_input(self):
        import json
        data = {
            "tldr": "test",
            "bull_case": ["a"],
            "bear_case": ["b"],
            "key_metrics_to_watch": ["c"],
            "qualitative_scores": {"moat": 5, "management": 5, "capital_allocation": 5, "industry_position": 5},
            "peer_comparison": "test",
            "verdict": "Watch",
            "confidence": 50,
            "sources_cited": [],
        }
        valid, report, _ = validate_report(json.dumps(data))
        assert valid

    def test_markdown_fenced_json(self):
        md = '```json\n{"tldr":"test","bull_case":["a"],"bear_case":["b"],"key_metrics_to_watch":["c"],"qualitative_scores":{"moat":5,"management":5,"capital_allocation":5,"industry_position":5},"peer_comparison":"test","verdict":"Watch","confidence":50,"sources_cited":[]}\n```'
        valid, report, _ = validate_report(md)
        assert valid

    def test_strong_buy_verdict(self):
        data = {
            "tldr": "test",
            "bull_case": ["a"],
            "bear_case": ["b"],
            "key_metrics_to_watch": ["c"],
            "qualitative_scores": {"moat": 5, "management": 5, "capital_allocation": 5, "industry_position": 5},
            "peer_comparison": "test",
            "verdict": "Strong Buy",
            "confidence": 90,
            "sources_cited": [],
        }
        valid, report, _ = validate_report(data)
        assert valid
        assert report.verdict == "Strong Buy"
