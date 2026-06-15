"""Multi-factor scoring pipeline.

Orchestrates: filters → factors → sector percentiles → HMM adjustment → composite score.

The pipeline:
1. Apply hard disqualifiers (flag, don't remove)
2. Compute all factors for each ticker
3. Rank each factor within GICS sector (percentiles)
4. Weight and aggregate into growth/quality/momentum subscores
5. Apply Sharpe ratio redistribution (if enabled)
6. Apply HMM regime bonus (on top N pre-scored stocks)
7. Output ranked DataFrame with all scores
"""

import pandas as pd
import numpy as np
from loguru import logger

from screen.filters import apply_disqualifiers
from screen.factors import compute_all_factors
from screen.hmm_detect import fit_regime_model, regime_adjustment


def score_universe(
    df: pd.DataFrame,
    config: dict | None = None,
    financials_map: dict[str, list[dict]] | None = None,
    price_map: dict[str, list[dict]] | None = None,
    log_returns_map: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Score the full universe of tickers.

    Args:
        df: DataFrame with columns matching fundamentals (market_cap, revenue_ttm, etc.)
            plus 'ticker' and 'sector'.
        config: Config dict (load from config.yaml if None).
        financials_map: Optional {ticker: [quarterly financials]} for multi-period factors.
        price_map: Optional {ticker: [daily price bars]} for momentum factors.
        log_returns_map: Optional {ticker: np.array of log returns} for HMM (avoids recompute).

    Returns:
        DataFrame with scores, ranks, and disqualification flags. Sorted by composite_score desc.
    """
    if config is None:
        from data.ingest import load_config
        config = load_config()

    financials_map = financials_map or {}
    price_map = price_map or {}
    log_returns_map = log_returns_map or {}

    scoring_cfg = config.get("scoring", {})
    weights_cfg = scoring_cfg.get("weights", {})
    sharpe_cfg = scoring_cfg.get("sharpe", {})
    hmm_cfg = scoring_cfg.get("hmm", {})

    # ── Step 1: Disqualifiers ─────────────────────────────────
    dq_cfg = config.get("disqualifiers", {})
    df = apply_disqualifiers(
        df,
        min_market_cap_b=dq_cfg.get("min_market_cap_b", 2.0),
        min_avg_volume_m=dq_cfg.get("min_avg_volume_m", 5.0),
        require_positive_fcf=dq_cfg.get("require_positive_fcf", True),
        max_debt_equity=dq_cfg.get("max_debt_equity", 3.0),
        min_altman_z=dq_cfg.get("min_altman_z", 1.8),
    )

    # ── Step 2: Compute factors ────────────────────────────────
    factor_names = [
        # Growth
        "revenue_yoy_ttm", "revenue_cagr_3yr", "eps_growth_ttm",
        "forward_revenue_growth", "reinvestment_rate", "rule_of_40",
        # Quality
        "roe_ttm", "roic", "gross_margin_ttm", "fcf_margin_ttm",
        "net_debt_ebitda", "accruals_ratio", "interest_coverage",
        # Momentum
        "return_6m_excl_1m", "return_12m", "return_1m",
        "earnings_surprise", "ma_ratio_50_200",
        # Risk-adjusted
        "sharpe_ratio",
    ]
    for name in factor_names:
        df[name] = 0.0

    for idx, row in df.iterrows():
        ticker = row["ticker"]
        fundamentals = row.to_dict()
        financials = financials_map.get(ticker, [])
        prices = price_map.get(ticker, [])

        factors = compute_all_factors(fundamentals, financials, prices, config)
        for k, v in factors.items():
            df.at[idx, k] = v

    # ── Step 3: Sector-relative percentiles ────────────────────
    def _pct_rank(series: pd.Series) -> pd.Series:
        """Percentile rank (0-100) within group, higher = better."""
        return series.rank(pct=True) * 100

    # For metrics where lower is better, invert before ranking
    invert_metrics = {"net_debt_ebitda", "accruals_ratio"}

    for name in factor_names:
        if name not in df.columns:
            continue
        if name in invert_metrics:
            df[f"{name}_pct"] = df.groupby("sector")[name].transform(
                lambda s: _pct_rank(-s)
            )
        else:
            df[f"{name}_pct"] = df.groupby("sector")[name].transform(_pct_rank)

    # ── Step 4: Weighted subscores ─────────────────────────────
    w_growth = weights_cfg.get("growth", 0.40)
    w_quality = weights_cfg.get("quality", 0.35)
    w_momentum = weights_cfg.get("momentum", 0.25)

    growth_metrics = scoring_cfg.get("growth", {}).get("metrics", {})
    quality_metrics = scoring_cfg.get("quality", {}).get("metrics", {})
    momentum_metrics = scoring_cfg.get("momentum", {}).get("metrics", {})

    df["growth_score"] = 0.0
    df["quality_score"] = 0.0
    df["momentum_score"] = 0.0

    for metric, weight in growth_metrics.items():
        col = f"{metric}_pct"
        if col in df.columns:
            df["growth_score"] += df[col] * weight

    for metric, weight in quality_metrics.items():
        col = f"{metric}_pct"
        if col in df.columns:
            df["quality_score"] += df[col] * weight

    # ── Sharpe integration into momentum ───────────────────────
    if sharpe_cfg.get("enabled", True):
        sharpe_weight = sharpe_cfg.get("weight_in_momentum", 0.20)
        # Redistribute: existing momentum metrics get reduced
        redistribution = 1.0 - sharpe_weight
        df["momentum_score"] = 0.0
        for metric, weight in momentum_metrics.items():
            col = f"{metric}_pct"
            if col in df.columns:
                df["momentum_score"] += df[col] * weight * redistribution
        # Add Sharpe
        if "sharpe_ratio_pct" in df.columns:
            df["momentum_score"] += df["sharpe_ratio_pct"] * sharpe_weight
    else:
        for metric, weight in momentum_metrics.items():
            col = f"{metric}_pct"
            if col in df.columns:
                df["momentum_score"] += df[col] * weight

    # ── Step 5: Composite pre-score ────────────────────────────
    df["composite_score"] = (
        df["growth_score"] * w_growth
        + df["quality_score"] * w_quality
        + df["momentum_score"] * w_momentum
    )

    # ── Step 6: HMM regime bonus ───────────────────────────────
    df["hmm_adjustment"] = 0.0
    df["hidden_gem_score"] = df["composite_score"].copy()

    if hmm_cfg.get("enabled", True) and log_returns_map:
        max_hmm = hmm_cfg.get("max_stocks_for_hmm", 200)

        # Pre-score ranking — only fit HMM on top N
        df_temp = df[~df["disqualified"]].copy()
        df_temp = df_temp.sort_values("composite_score", ascending=False)

        top_tickers = df_temp.head(max_hmm)["ticker"].tolist()
        logger.info(f"Fitting HMM on top {len(top_tickers)} pre-scored stocks...")

        for ticker in top_tickers:
            lr = log_returns_map.get(ticker)
            if lr is None or len(lr) < 50:
                continue

            try:
                hmm_result = fit_regime_model(
                    lr,
                    n_components=hmm_cfg.get("n_components", 3),
                    n_seeds=hmm_cfg.get("n_seeds", 20),
                    n_iter=hmm_cfg.get("n_iter", 500),
                )
                bonus = regime_adjustment(
                    hmm_result,
                    max_bonus=hmm_cfg.get("regime_bonus", 0.10),
                )
                mask = df["ticker"] == ticker
                df.loc[mask, "hmm_adjustment"] = bonus
                df.loc[mask, "composite_score"] += bonus * 10  # scale to 0-100 range

                # Hidden gem score = composite + Sharpe + HMM excess
                sharpe_pct = df.loc[mask, "sharpe_ratio_pct"].values[0] if "sharpe_ratio_pct" in df.columns else 50
                hmm_excess_pct = min(100, max(0, 50 + hmm_result.get("actual_excess_return", 0) * 100))
                df.loc[mask, "hidden_gem_score"] = (
                    df.loc[mask, "composite_score"].values[0]
                    + sharpe_pct * 0.1
                    + hmm_excess_pct * 0.1
                )
            except Exception as e:
                logger.warning(f"HMM failed for {ticker}: {e}")

    # ── Step 7: Rank and sort ──────────────────────────────────
    df["rank"] = df["composite_score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    n_disqualified = df["disqualified"].sum()
    n_passing = len(df) - n_disqualified
    logger.info(
        f"Scoring complete: {n_disqualified} disqualified, "
        f"{n_passing} scored. Top 5: {df.head(5)['ticker'].tolist()}"
    )

    return df
