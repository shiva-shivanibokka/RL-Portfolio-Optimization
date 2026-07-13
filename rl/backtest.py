"""Backtest engine and baselines. numpy + pandas only (no torch)."""
from typing import Callable
import numpy as np
import pandas as pd

from rl import metrics

Policy = Callable[[np.ndarray], np.ndarray]


def build_observation(returns_window: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(returns_window, float).ravel(),
                           np.asarray(current_weights, float).ravel()])


def _rebalance_mask(index: pd.DatetimeIndex, rebalance: str) -> np.ndarray:
    """True on days a rebalance happens. 'D' every day; 'W'/'M' on first bday of period."""
    if rebalance == "D":
        return np.ones(len(index), dtype=bool)
    period = {"W": "W", "M": "M"}[rebalance]
    grp = index.to_period(period)
    mask = np.zeros(len(index), dtype=bool)
    _, first_idx = np.unique(grp, return_index=True)
    mask[first_idx] = True
    return mask


def _walk(prices: pd.DataFrame, weight_fn, *, lookback: int, cost: float, rebalance: str) -> dict:
    simple = prices.pct_change().fillna(0.0)
    dates = prices.index
    n_assets = prices.shape[1]
    start = max(lookback, 1)

    reb = _rebalance_mask(dates, rebalance)
    weights = np.full(n_assets, 1.0 / n_assets)
    equity = [1.0]
    port_returns = []
    timeline = []
    turnovers = []

    for t in range(start, len(dates)):
        if reb[t]:
            window = simple.values[t - lookback:t]
            new_w = np.asarray(weight_fn(window, weights), float)
            new_w = np.clip(new_w, 0.0, None)
            s = new_w.sum()
            new_w = new_w / s if s > 0 else np.full(n_assets, 1.0 / n_assets)
            turn = np.abs(new_w - weights).sum()
            turnovers.append(turn)
            weights = new_w
            timeline.append({"date": dates[t].date().isoformat(),
                             "weights": weights.tolist()})
        else:
            turn = 0.0
        gross = float(1.0 + (simple.values[t] * weights).sum())
        net = gross * (1.0 - cost * turn)
        equity.append(equity[-1] * net)
        port_returns.append(net - 1.0)

    # NOTE (deviation from brief): the brief's Step 3 dropped the leading 1.0
    # baseline entry (`equity[1:]`) and then renormalized by the new first
    # entry (`equity / equity[0]`). That silently discards the first traded
    # day's return from total_return/cagr (verified: it changes the number,
    # not just cosmetic). `equity[0]` is already exactly 1.0 by construction
    # (the pre-trade baseline), so no renormalization is needed -- we just
    # keep the full list. `dates` is aligned to include that baseline date
    # (the day before the first trade) so dates and equity stay same length.
    equity = np.array(equity)
    ret = np.array(port_returns)
    out_dates = [d.date().isoformat() for d in dates[start - 1:]]
    return {
        "dates": out_dates,
        "equity": equity.tolist(),
        "weights_timeline": timeline,
        "metrics": {
            "total_return": metrics.total_return(equity),
            "cagr": metrics.cagr(equity),
            "annual_vol": metrics.annual_vol(ret),
            "sharpe": metrics.sharpe(ret),
            "max_drawdown": metrics.max_drawdown(equity),
            "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        },
    }


def run_backtest(prices: pd.DataFrame, policy: Policy, *, lookback: int,
                 rebalance: str, cost: float = 0.001) -> dict:
    def weight_fn(window, current):
        obs = build_observation(window, current)
        return policy(obs)
    return _walk(prices, weight_fn, lookback=lookback, cost=cost, rebalance=rebalance)


def equal_weight_backtest(prices: pd.DataFrame, *, rebalance: str = "M", cost: float = 0.001) -> dict:
    n = prices.shape[1]
    def weight_fn(window, current):
        return np.full(n, 1.0 / n)
    return _walk(prices, weight_fn, lookback=1, cost=cost, rebalance=rebalance)


def buy_and_hold(prices: pd.DataFrame, ticker: str) -> dict:
    n = prices.shape[1]
    col = list(prices.columns).index(ticker)
    onehot = np.zeros(n); onehot[col] = 1.0
    def weight_fn(window, current):
        return onehot
    return _walk(prices, weight_fn, lookback=1, cost=0.0, rebalance="D")
