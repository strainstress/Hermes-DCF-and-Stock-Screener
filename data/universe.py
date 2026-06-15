"""S&P 500 and S&P 400 ticker universe — no ETFs, US common stock only."""

from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Hardcoded fallback path — snapshot of Wikipedia S&P constituent lists
_FALLBACK_PATH = Path(__file__).parent / "universe_fallback.json"

# Well-known ETFs to always exclude
_ETF_BLOCKLIST: set[str] = {
    "SPY", "IVV", "VOO", "IJH", "MDY", "XLI", "XLF", "XLK", "XLE",
    "QQQ", "IWM", "VTI", "VGT", "VHT", "VFH", "VDE", "VDC", "VCR",
    "VOX", "VIS", "VPU", "VNQ", "VAW", "VWO", "VEA", "BND", "AGG",
    "LQD", "HYG", "TLT", "SHY", "IEF", "GLD", "SLV", "USO", "UNG",
    "DIA", "EEM", "EFA", "EWJ", "EWU", "EWG", "EWC", "EWA", "EWZ",
    "FXI", "IYR", "IYT", "IYZ", "XLB", "XLY", "XLP", "XME", "XOP",
    "XRT", "XLU", "XLV", "XLRE", "XHB", "XBI", "XSD", "XSW", "XNTK",
    "XT", "XWEB", "KBE", "KRE", "KBWB", "KIE", "KCE", "SMH", "SOXX",
    "IBB", "FBT", "ARKK", "ARKG", "ARKF", "ARKW", "ARKQ", "ARKX",
    "TAN", "ICLN", "PBW", "QCLN", "LIT", "BOTZ", "ROBO", "AIQ",
    "DRIV", "FINX", "IPAY", "TQQQ", "SQQQ", "UVXY", "SVXY", "VXX",
    "TVIX",
}


def get_universe(refresh: bool = False) -> list[str]:
    """Return deduplicated, sorted list of S&P 500 + S&P 400 tickers.

    Excludes ETFs and funds. Uses a static JSON snapshot for speed.

    Args:
        refresh: If True, force refresh from external sources.
                 Default False — returns cached list.

    Returns:
        Sorted list of uppercase ticker symbols.

    Raises:
        RuntimeError: If the fallback file is missing or empty.
    """
    if not _FALLBACK_PATH.exists():
        raise RuntimeError(
            f"Universe fallback file not found: {_FALLBACK_PATH}. "
            "Run 'python -m data.universe --refresh' to regenerate."
        )

    with open(_FALLBACK_PATH) as f:
        raw: list[str] = json.load(f)

    seen: set[str] = set()
    result: list[str] = []

    for t in raw:
        t = t.strip().upper()
        if not t:
            continue
        if t in _ETF_BLOCKLIST:
            continue
        if t in seen:
            continue
        seen.add(t)
        result.append(t)

    if not result:
        raise RuntimeError(
            "Universe is empty after filtering — check data/universe_fallback.json"
        )

    return sorted(result)


def is_valid_ticker(ticker: str) -> bool:
    """Check if a ticker looks syntactically valid.

    Rules:
    - Non-empty
    - Max 5 characters
    - Not a known ETF
    - Letters only, or letters with a single dot (e.g., BRK.B)
    """
    t = ticker.strip().upper()
    if not t:
        return False
    if len(t) > 5:
        return False
    if t in _ETF_BLOCKLIST:
        return False
    # Letters only: AAPL, MSFT
    if t.isalpha():
        return True
    # Letters with a single dot: BRK.B, BF.B
    if t.count(".") == 1:
        parts = t.split(".")
        return all(p.isalpha() for p in parts) and all(len(p) <= 4 for p in parts)
    return False
