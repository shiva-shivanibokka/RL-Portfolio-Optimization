import numpy as np
from rl.env import PortfolioEnv


def _returns(t=600, n=9):
    rng = np.random.default_rng(0)
    return rng.normal(0.0003, 0.01, size=(t, n))


def test_softmax_weights_valid():
    w = PortfolioEnv.weights_from_action(np.array([1.0, 2.0, 3.0]))
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()


def test_reset_returns_obs_of_right_shape():
    env = PortfolioEnv(_returns(), lookback=30, episode_len=100,
                       lambda_risk=0.1, lambda_cost=0.1)
    obs, info = env.reset(seed=1)
    assert obs.shape == (30 * 9 + 9,)


def test_step_returns_gym_tuple_and_terminates():
    env = PortfolioEnv(_returns(), lookback=30, episode_len=5,
                       lambda_risk=0.1, lambda_cost=0.1)
    env.reset(seed=1)
    done = False
    steps = 0
    while not done and steps < 10:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert steps <= 6
    assert isinstance(reward, float)


def test_higher_cost_lambda_penalizes_turnover():
    r = _returns()
    cheap = PortfolioEnv(r, lookback=30, episode_len=50, lambda_risk=0.0, lambda_cost=0.0)
    pricey = PortfolioEnv(r, lookback=30, episode_len=50, lambda_risk=0.0, lambda_cost=5.0)
    cheap.reset(seed=2); pricey.reset(seed=2)
    a1 = np.zeros(9); a2 = np.zeros(9); a2[0] = 10.0  # a2 forces a big weight shift
    cheap.step(a1); pricey.step(a1)
    _, r_cheap, *_ = cheap.step(a2)
    _, r_pricey, *_ = pricey.step(a2)
    assert r_pricey < r_cheap


def test_higher_risk_lambda_lowers_reward():
    r = _returns()
    norisk = PortfolioEnv(r, lookback=30, episode_len=50, lambda_risk=0.0, lambda_cost=0.0)
    risky = PortfolioEnv(r, lookback=30, episode_len=50, lambda_risk=50.0, lambda_cost=0.0)
    norisk.reset(seed=3); risky.reset(seed=3)
    a = np.zeros(9); a[0] = 10.0  # concentrate -> nonzero ex-ante vol
    _, r_norisk, *_ = norisk.step(a)
    _, r_risky, *_ = risky.step(a)
    assert r_risky < r_norisk  # identical return/turnover; only the risk penalty differs


def test_portfolio_vol_favors_diversification():
    # Core property of the ex-ante risk term: the agent can lower it by
    # diversifying (or tilting to low-vol assets), so lambda_risk creates a
    # real gradient. Equal-weight must have lower ex-ante vol than an all-in bet.
    env = PortfolioEnv(_returns(), lookback=30, episode_len=50,
                       lambda_risk=1.0, lambda_cost=0.0)
    env.reset(seed=4)
    concentrated = PortfolioEnv.weights_from_action(np.array([10.0] + [0.0] * 8))
    diversified = PortfolioEnv.weights_from_action(np.zeros(9))
    assert env._portfolio_vol(diversified) < env._portfolio_vol(concentrated)
