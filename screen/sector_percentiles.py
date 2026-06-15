"""Sector-relative percentile ranking for multi-factor scoring.

All metric ranking happens within GICS sector — a 30% gross margin
in software is mediocre; in retail it's elite.

This module is used by screen/score.py for the groupby percentile transform.
"""

import pandas as pd


def sector_percentile(
    df: pd.DataFrame,
    metric: str,
    sector_col: str = "sector",
    higher_is_better: bool = True,
) -> pd.Series:
    """Compute percentile rank of a metric within each sector.

    Args:
        df: DataFrame with ticker data.
        metric: Column name to rank.
        sector_col: Column name for sector grouping.
        higher_is_better: If True, higher values rank higher.
                          If False, the metric is inverted before ranking.

    Returns:
        Series with percentile ranks (0-100).
    """
    if metric not in df.columns:
        return pd.Series(0.0, index=df.index)

    if higher_is_better:
        return df.groupby(sector_col)[metric].transform(
            lambda s: s.rank(pct=True) * 100
        )
    else:
        return df.groupby(sector_col)[metric].transform(
            lambda s: (-s).rank(pct=True) * 100
        )
