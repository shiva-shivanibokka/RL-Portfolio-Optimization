import numpy as np
from rl.metrics import total_return, cagr, annual_vol, sharpe, max_drawdown


def test_total_return():
    equity = np.array([1.0, 1.1, 1.21])
    assert abs(total_return(equity) - 0.21) < 1e-9


def test_max_drawdown_known_series():
    # peak 1.0 -> trough 0.75 -> recover: max DD = -0.25
    equity = np.array([1.0, 0.9, 0.75, 0.8, 1.0])
    assert abs(max_drawdown(equity) - (-0.25)) < 1e-9


def test_annual_vol_constant_returns_is_zero():
    returns = np.array([0.001, 0.001, 0.001, 0.001])
    assert abs(annual_vol(returns)) < 1e-12


def test_sharpe_zero_vol_is_zero_not_nan():
    returns = np.array([0.0, 0.0, 0.0])
    assert sharpe(returns) == 0.0


def test_cagr_doubles_in_one_year():
    equity = np.concatenate([np.linspace(1.0, 2.0, 252)])
    assert abs(cagr(equity) - 1.0) < 0.02
