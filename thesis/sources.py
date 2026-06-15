"""SEC EDGAR filing fetcher — 10-K and 10-Q documents.

Uses the SEC's EDGAR API (free, no key required).
Fetches full filing text, caches locally.
"""

import time
import requests
from pathlib import Path
from loguru import logger

# SEC requires a user agent identifying the caller
SEC_USER_AGENT = "stock-screener/0.1.0 (contact@example.com)"
SEC_BASE = "https://www.sec.gov"


def _sec_headers() -> dict:
    return {"User-Agent": SEC_USER_AGENT, "Accept": "application/json"}


def get_company_filings(cik: str, form_types: list[str] | None = None) -> list[dict]:
    """Get recent filings for a company by CIK number.

    Args:
        cik: 10-digit CIK number (zero-padded, e.g., '0000320193' for AAPL).
        form_types: List of form types to filter (e.g., ['10-K', '10-Q']).
                   Default: ['10-K', '10-Q'].

    Returns:
        List of filing dicts with keys: accessionNumber, form, filingDate, primaryDocument.
    """
    if form_types is None:
        form_types = ["10-K", "10-Q"]

    url = f"{SEC_BASE}/cgi-bin/browse-edgar"
    params = {
        "action": "getcompany",
        "CIK": cik.lstrip("0"),
        "type": ",".join(form_types),
        "dateb": "",
        "owner": "exclude",
        "count": 10,
        "output": "json",
    }

    try:
        resp = requests.get(url, headers=_sec_headers(), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("filings", {}).get("recent", {}).get("accessionNumber", []) or []
    except Exception as e:
        logger.error(f"Failed to get filings for CIK {cik}: {e}")
        return []


def fetch_filing_text(cik: str, accession_number: str, primary_document: str) -> str:
    """Fetch the full text of a specific SEC filing.

    Args:
        cik: 10-digit CIK number (zero-padded).
        accession_number: Filing accession number (dashes removed for URL).
        primary_document: Primary document filename (e.g., 'aapl-20250927.htm').

    Returns:
        Full filing text, or empty string on failure.
    """
    # EDGAR URL format: /Archives/edgar/data/CIK/accession_clean/primary_document
    acc_clean = accession_number.replace("-", "")
    url = f"{SEC_BASE}/Archives/edgar/data/{cik.lstrip('0')}/{acc_clean}/{primary_document}"

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": SEC_USER_AGENT},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error(f"Failed to fetch filing {accession_number}: {e}")
        return ""


def fetch_latest_10k(cik: str) -> str:
    """Fetch the most recent 10-K filing text for a company.

    Returns full 10-K text, or empty string if not found.
    """
    filings = get_company_filings(cik, form_types=["10-K"])
    if not filings:
        return ""
    # Filings are returned most-recent-first
    acc = filings[0]
    return fetch_filing_text(cik, acc.get("accessionNumber", ""), acc.get("primaryDocument", ""))


def fetch_latest_10q(cik: str) -> str:
    """Fetch the most recent 10-Q filing text for a company."""
    filings = get_company_filings(cik, form_types=["10-Q"])
    if not filings:
        return ""
    acc = filings[0]
    return fetch_filing_text(cik, acc.get("accessionNumber", ""), acc.get("primaryDocument", ""))


# ── Excerpt extraction (avoid sending 200KB of HTML to the LLM) ──


def extract_relevant_excerpts(filing_text: str, max_chars: int = 15_000) -> str:
    """Extract the most relevant sections from a 10-K/10-Q filing.

    Strips HTML tags and focuses on:
    - Business description (Item 1)
    - Risk factors (Item 1A)
    - MD&A (Item 7)
    - Financial statements summary

    Simple approach for v1: strip HTML, take first max_chars chars
    after finding key section markers.

    Args:
        filing_text: Raw HTML/text of the filing.
        max_chars: Maximum characters to return.

    Returns:
        Excerpted text focusing on key sections.
    """
    if not filing_text:
        return ""

    # Basic HTML stripping
    import re
    text = re.sub(r"<[^>]+>", " ", filing_text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Try to find key sections
    sections = []
    markers = [
        ("Item 1.", "Business"),
        ("Item 1A.", "Risk Factors"),
        ("Item 7.", "Management's Discussion"),
        ("ITEM 1.", "BUSINESS"),
        ("ITEM 1A.", "RISK FACTORS"),
        ("ITEM 7.", "MANAGEMENT"),
    ]

    for marker, _label in markers:
        idx = text.find(marker)
        if idx >= 0:
            # Take 3000 chars from each found section
            section_text = text[idx:idx + 3000]
            sections.append(section_text)

    if sections:
        result = "\n\n---\n\n".join(sections)
        return result[:max_chars]

    # Fallback: just return the first max_chars
    return text[:max_chars]


# ── CIK Lookup ────────────────────────────────────────────────


def ticker_to_cik(ticker: str) -> str:
    """Convert a ticker symbol to a 10-digit CIK number.

    Uses the SEC's company_tickers.json endpoint.

    Args:
        ticker: Uppercase ticker symbol.

    Returns:
        10-digit zero-padded CIK string, or empty string if not found.
    """
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=_sec_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"])
                return cik.zfill(10)

        logger.warning(f"CIK not found for ticker {ticker}")
        return ""
    except Exception as e:
        logger.error(f"CIK lookup failed: {e}")
        return ""
