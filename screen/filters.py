"""Hard disqualifier filters for the stock screener.

A stock must clear ALL disqualifiers to advance to scoring.
Fail any one → flagged as disqualified with a reason.
"""

import pandas as pd
from loguru import logger


def apply_disqualifiers(
    df: pd.DataFrame,
    min_market_cap_b: float = 2.0,
    min_avg_volume_m: float = 5.0,
    require_positive_fcf: bool = True,
    max_debt_equity: float = 3.0,
    min_altman_z: float = 1.8,
) -> pd.DataFrame:
    """Apply hard disqualifiers to a DataFrame of ticker data.

    Adds two columns:
        disqualified: bool — True if any disqualifier triggers
        disqualifier_reason: str — reason for disqualification

    Args:
        df: DataFrame with columns market_cap, avg_volume_20d, fcf_ttm,
            debt_equity, altman_z.
        min_market_cap_b: Minimum market cap in billions.
        min_avg_volume_m: Minimum 20-day average volume in millions.
        require_positive_fcf: If True, disqualify negative TTM FCF.
        max_debt_equity: Maximum debt-to-equity ratio.
        min_altman_z: Minimum Altman Z-score.

    Returns:
        DataFrame with added 'disqualified' and 'disqualifier_reason' columns.
    """
    df = df.copy()
    df["disqualified"] = False
    df["disqualifier_reason"] = ""

    # ── Market cap ─────────────────────────────────────────────
    mask = df["market_cap"] < (min_market_cap_b * 1e9)
    if mask.any():
        applicable = mask & ~df["disqualified"]
        df.loc[applicable, "disqualifier_reason"] = (
            f"Market cap below ${min_market_cap_b:.0f}B threshold"
        )
        df.loc[applicable, "disqualified"] = True

    # ── Average volume ─────────────────────────────────────────
    mask = df.get("avg_volume_20d", pd.Series(0, index=df.index)) < (min_avg_volume_m * 1e6)
    if mask.any():
        applicable = mask & ~df["disqualified"]
        df.loc[applicable, "disqualifier_reason"] = (
            f"Average volume below ${min_avg_volume_m:.0f}M threshold"
        )
        df.loc[applicable, "disqualified"] = True

    # ── Negative free cash flow ────────────────────────────────
    if require_positive_fcf:
        mask = df.get("fcf_ttm", pd.Series(0, index=df.index)) < 0
        if mask.any():
            applicable = mask & ~df["disqualified"]
            df.loc[applicable, "disqualifier_reason"] = "Negative TTM free cash flow"
            df.loc[applicable, "disqualified"] = True

    # ── Debt-to-equity ─────────────────────────────────────────
    mask = df.get("debt_equity", pd.Series(0, index=df.index)) > max_debt_equity
    if mask.any():
        applicable = mask & ~df["disqualified"]
        df.loc[applicable, "disqualifier_reason"] = (
            f"Debt/Equity ratio > {max_debt_equity:.1f}"
        )
        df.loc[applicable, "disqualified"] = True

    # ── Altman Z-score ─────────────────────────────────────────
    mask = df.get("altman_z", pd.Series(99, index=df.index)) < min_altman_z
    if mask.any():
        applicable = mask & ~df["disqualified"]
        df.loc[applicable, "disqualifier_reason"] = (
            f"Altman Z-score < {min_altman_z}"
        )
        df.loc[applicable, "disqualified"] = True

    n_disqualified = df["disqualified"].sum()
    pct = (n_disqualified / len(df) * 100) if len(df) > 0 else 0.0
    logger.info(
        f"Disqualifiers: {n_disqualified}/{len(df)} flagged "
        f"({pct:.1f}%)"
    )
    return df
