"""Custom Gymnasium portfolio environment. Training-time only (imports gymnasium)."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl.backtest import build_observation


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, simple_returns, *, lookback=30, episode_len=252,
                 lambda_risk, lambda_cost, cost=0.001, seed=None):
        super().__init__()
        self.returns = np.asarray(simple_returns, dtype=np.float32)
        self.T, self.N = self.returns.shape
        self.lookback = lookback
        self.episode_len = episode_len
        self.lambda_risk = lambda_risk
        self.lambda_cost = lambda_cost
        self.cost = cost
        self._rng = np.random.default_rng(seed)

        obs_dim = lookback * self.N + self.N
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-10.0, 10.0, shape=(self.N,), dtype=np.float32)
        self._start = None
        self._t = None
        self._weights = None

    @staticmethod
    def weights_from_action(action):
        a = np.asarray(action, dtype=float)
        a = a - a.max()
        e = np.exp(a)
        return e / e.sum()

    def _obs(self):
        window = self.returns[self._t - self.lookback:self._t]
        return build_observation(window, self._weights).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        high = self.T - self.episode_len - 1
        self._start = int(self._rng.integers(self.lookback, max(self.lookback + 1, high)))
        self._t = self._start
        self._weights = np.full(self.N, 1.0 / self.N)
        return self._obs(), {}

    def _portfolio_vol(self, weights):
        """Ex-ante Markowitz portfolio volatility from the trailing-window
        covariance: sqrt(wT Sigma w). This is directly controllable by the
        agent's weights (unlike realized rolling variance), so lambda_risk
        produces a real gradient toward low-vol/low-covariance assets."""
        window = self.returns[self._t - self.lookback:self._t]
        cov = np.cov(window, rowvar=False)
        return float(np.sqrt(max(float(weights @ cov @ weights), 0.0)))

    def step(self, action):
        new_w = self.weights_from_action(action)
        turnover = float(np.abs(new_w - self._weights).sum())
        port_ret = float((self.returns[self._t] * new_w).sum())
        port_vol = self._portfolio_vol(new_w)

        reward = np.log1p(port_ret) - self.lambda_risk * port_vol - self.lambda_cost * self.cost * turnover

        self._weights = new_w
        self._t += 1
        terminated = False
        truncated = (self._t - self._start) >= self.episode_len or self._t >= self.T - 1
        return self._obs(), float(reward), terminated, truncated, {}
