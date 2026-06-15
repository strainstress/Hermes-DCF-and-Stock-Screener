"""Hidden Markov Model regime detection for stocks.

Fits a 3-state Gaussian HMM on 2-year log returns to identify
Bull / Sideways / Bear regimes. Used to surface "hidden gems" —
stocks outperforming their peers in the same regime.

Reference: quantitative-stock-analysis skill (hmmlearn, 20 seeds, best-score).
"""

import numpy as np
from hmmlearn import hmm
from loguru import logger


def fit_regime_model(
    log_returns: np.ndarray,
    n_components: int = 3,
    n_seeds: int = 20,
    n_iter: int = 500,
) -> dict:
    """Fit a Gaussian HMM on log returns to detect market regimes.

    Tries multiple random seeds and picks the best-scoring model.

    Args:
        log_returns: 1D array of log returns (daily).
        n_components: Number of hidden states (default 3: Bear/Sideways/Bull).
        n_seeds: Random seeds to try for convergence.
        n_iter: EM iterations per fit.

    Returns:
        Dict with regime labels, means, vols, transition matrix,
        current state probabilities, and bull probability.
    """
    if len(log_returns) < 50:
        logger.warning("Insufficient data for HMM — returning neutral")
        return _neutral_result()

    X = log_returns.reshape(-1, 1)

    best_score = -np.inf
    best_model = None

    for seed in range(n_seeds):
        try:
            m = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type="diag",
                n_iter=n_iter,
                random_state=seed,
                tol=1e-4,
            )
            m.fit(X)
            score = m.score(X)
            if score > best_score:
                best_score = score
                best_model = m
        except Exception:
            continue

    if best_model is None:
        logger.warning("HMM failed to converge — returning neutral")
        return _neutral_result()

    model = best_model
    means = model.means_.flatten()
    vols = np.sqrt(model.covars_.flatten())
    transmat = model.transmat_

    # Map states to Bear (0), Sideways (1), Bull (2) by mean return
    regime_order = np.argsort(means)
    labels = {
        regime_order[0]: "BEAR",
        regime_order[1]: "SIDEWAYS",
        regime_order[2]: "BULL",
    }

    # Annualize
    ann_means = means * 252
    ann_vols = vols * np.sqrt(252)

    # Current state and probabilities
    states = model.predict(X)
    probs = model.predict_proba(X)
    current_state = states[-1]
    current_probs = probs[-1]

    # Reorder probabilities to Bear, Sideways, Bull
    ordered_probs = {
        "bear": current_probs[regime_order[0]],
        "sideways": current_probs[regime_order[1]],
        "bull": current_probs[regime_order[2]],
    }

    # Stickiness (diagonal of transition matrix)
    stickiness = {
        labels[i]: transmat[i, i]
        for i in range(n_components)
    }

    # Expected annual return for current regime
    regime_annual_return = ann_means[current_state]

    # Actual annualized return over the full period
    actual_annual_return = np.mean(log_returns) * 252

    # Excess: how much the stock is beating (or missing) its current regime
    actual_excess_return = actual_annual_return - regime_annual_return

    # Bull probability = P(current state is BULL)
    bull_probability = current_probs[regime_order[2]]

    # Check for collapsed regimes (fewer than 3 effective states)
    unique_states = len(set(states))
    if unique_states < 3:
        logger.debug(
            f"HMM collapsed to {unique_states} effective states "
            f"(persistent trend may cause this — treat as signal)"
        )

    return {
        "means": ann_means[regime_order].tolist(),
        "vols": ann_vols[regime_order].tolist(),
        "regime_labels": {i: labels[i] for i in range(n_components)},
        "current_state": current_state,
        "current_state_label": labels[current_state],
        "current_probs": [
            current_probs[regime_order[0]],
            current_probs[regime_order[1]],
            current_probs[regime_order[2]],
        ],
        "ordered_probs": ordered_probs,
        "transition_matrix": transmat.tolist(),
        "bull_probability": float(bull_probability),
        "regime_stickiness": stickiness,
        "regime_annual_return": float(regime_annual_return),
        "actual_excess_return": float(actual_excess_return),
        "n_effective_states": unique_states,
    }


def regime_adjustment(
    hmm_result: dict,
    max_bonus: float = 0.10,
    max_penalty: float = 0.10,
) -> float:
    """Convert HMM regime data into a scoring adjustment.

    Logic:
    - Stock beating its regime expectations → positive bonus
    - Stock meeting expectations → neutral (~0)
    - Stock underperforming its regime → negative penalty

    The adjustment is scaled: a stock in Bear regime returning +10%
    gets a bigger bonus than a stock in Bull regime returning +10%.

    Args:
        hmm_result: Dict from fit_regime_model().
        max_bonus: Maximum positive adjustment.
        max_penalty: Maximum negative adjustment.

    Returns:
        Float in [-max_penalty, +max_bonus].
    """
    excess = hmm_result.get("actual_excess_return", 0.0)
    bull_prob = hmm_result.get("bull_probability", 0.5)

    # The adjustment = excess_return weighted by how unexpected it is.
    # In Bear (bull_prob ~ 0): exceeding expectations is very impressive.
    # In Bull (bull_prob ~ 1): you're expected to do well, excess is less meaningful.
    surprise_factor = 1.0 - bull_prob

    # Scale: +10% excess in Bear (surprise=0.95) → ~0.095 bonus
    #         +10% excess in Bull (surprise=0.05) → ~0.005 bonus
    raw_adjustment = excess * surprise_factor * 0.5

    # Clamp
    return float(np.clip(raw_adjustment, -max_penalty, max_bonus))


def _neutral_result() -> dict:
    """Return a neutral HMM result when fitting fails or data is insufficient."""
    return {
        "means": [0.0, 0.0, 0.0],
        "vols": [0.0, 0.0, 0.0],
        "regime_labels": {0: "BEAR", 1: "SIDEWAYS", 2: "BULL"},
        "current_state": 1,
        "current_state_label": "SIDEWAYS",
        "current_probs": [0.33, 0.34, 0.33],
        "ordered_probs": {"bear": 0.33, "sideways": 0.34, "bull": 0.33},
        "transition_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "bull_probability": 0.33,
        "regime_stickiness": {"BEAR": 1.0, "SIDEWAYS": 1.0, "BULL": 1.0},
        "regime_annual_return": 0.0,
        "actual_excess_return": 0.0,
        "n_effective_states": 3,
    }
