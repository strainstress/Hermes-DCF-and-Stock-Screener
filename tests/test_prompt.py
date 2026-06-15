"""Tests for thesis/prompt.py."""
from thesis.prompt import build_user_prompt, build_messages, SYSTEM_PROMPT


def test_build_user_prompt_includes_ticker():
    prompt = build_user_prompt(
        ticker="AAPL", name="Apple Inc.", sector="Technology",
        market_cap=3e12, description="Makes iPhones",
        ten_k_excerpt="Item 1. Business...",
        ten_q_excerpt="Item 2. Financials...",
        transcripts="Q3: Revenue up 5%",
        recent_news="Apple announces new product",
        peer_financials="MSFT: 40% margin",
        growth_score=75, quality_score=80, momentum_score=65,
        composite_score=73, sharpe_ratio=1.2, hmm_regime="BULL",
    )
    assert "AAPL" in prompt
    assert "Apple Inc." in prompt
    assert "Technology" in prompt
    assert "Item 1. Business" in prompt
    assert "Strong Buy | Buy | Watch | Pass | Sell" in prompt


def test_build_user_prompt_handles_missing_sources():
    prompt = build_user_prompt(
        ticker="TEST", name="Test", sector="Unknown",
        market_cap=1e9, description="Test company",
        ten_k_excerpt="", ten_q_excerpt="",
        transcripts="", recent_news="", peer_financials="",
        growth_score=0, quality_score=0, momentum_score=0,
        composite_score=0, sharpe_ratio=0, hmm_regime="Unknown",
    )
    assert "Not available" in prompt
    assert "TEST" in prompt


def test_build_messages_returns_list_of_dicts():
    msgs = build_messages(
        ticker="AAPL", name="Apple", sector="Tech",
        market_cap=3e12, description="iPhone maker",
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "buy-side equity analyst" in msgs[0]["content"]


def test_system_prompt_contains_key_directives():
    assert "buy-side equity analyst" in SYSTEM_PROMPT
    assert "Be specific" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT
