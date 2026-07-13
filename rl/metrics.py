"""Pure financial metrics. numpy only."""
import numpy as np


def total_return(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    return float(equity[-1] / equity[0] - 1.0)


def cagr(equity: np.ndarray, periods_per_year: int = 252) -> float:
    equity = np.asarray(equity, dtype=float)
    n = len(equity) - 1
    if n <= 0:
        return 0.0
    years = n / periods_per_year
    return float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)


def annual_vol(returns: np.ndarray, periods_per_year: int = 252) -> float:
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    returns = np.asarray(returns, dtype=float)
    vol = returns.std(ddof=1) if returns.size >= 2 else 0.0
    if vol == 0.0:
        return 0.0
    return float(returns.mean() / vol * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return float(drawdowns.min())
