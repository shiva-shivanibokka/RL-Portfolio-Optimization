import numpy as np
import pandas as pd
from rl.backtest import run_backtest, equal_weight_backtest, buy_and_hold, build_observation


def _prices(n=80, cols=("A", "B")):
    idx = pd.bdate_range("2020-01-01", periods=n)
    # A rises 0.1%/day, B flat
    a = 100 * (1.001 ** np.arange(n))
    b = np.full(n, 50.0)
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_build_observation_shape():
    window = np.zeros((30, 9))
    weights = np.zeros(9)
    obs = build_observation(window, weights)
    assert obs.shape == (30 * 9 + 9,)


def test_buy_and_hold_matches_price_move():
    prices = _prices()
    res = buy_and_hold(prices, "A")
    expected = prices["A"].iloc[-1] / prices["A"].iloc[0] - 1.0
    assert abs(res["metrics"]["total_return"] - expected) < 1e-9


def test_all_in_best_asset_beats_equal_weight():
    prices = _prices()

    def all_A(obs):
        return np.array([1.0, 0.0])

    agent = run_backtest(prices, all_A, lookback=10, rebalance="D", cost=0.0)
    ew = equal_weight_backtest(prices, rebalance="M", cost=0.0)
    assert agent["metrics"]["total_return"] > ew["metrics"]["total_return"]


def test_weights_sum_to_one_each_rebalance():
    prices = _prices()

    def half(obs):
        return np.array([0.5, 0.5])

    res = run_backtest(prices, half, lookback=10, rebalance="W", cost=0.0)
    for row in res["weights_timeline"]:
        assert abs(sum(row["weights"]) - 1.0) < 1e-9


def test_turnover_cost_reduces_return():
    prices = _prices()
    flip = {"i": 0}

    def flipper(obs):
        flip["i"] += 1
        return np.array([1.0, 0.0]) if flip["i"] % 2 else np.array([0.0, 1.0])

    no_cost = run_backtest(prices, flipper, lookback=10, rebalance="D", cost=0.0)
    with_cost = run_backtest(prices, flipper, lookback=10, rebalance="D", cost=0.01)
    assert with_cost["metrics"]["total_return"] < no_cost["metrics"]["total_return"]
