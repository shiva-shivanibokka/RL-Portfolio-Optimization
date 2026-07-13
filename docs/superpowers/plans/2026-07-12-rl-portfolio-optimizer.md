# RL Portfolio Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive web app where a visitor picks a strategy profile and date window and sees a trained PPO portfolio agent backtested live against baselines, deployed on Vercel free tier.

**Architecture:** Python (offline) trains three reward-shaped PPO agents on a real ETF price snapshot and exports each policy's MLP weights to `.npz`. A pure-numpy backtest engine + numpy policy inference power a Vercel Python serverless function (`/api/backtest`). A Next.js App Router frontend (repo root) calls that function and renders charts. torch/SB3 are training-time only and never shipped.

**Tech Stack:** Python 3.12, Gymnasium, Stable-Baselines3 (PPO), PyTorch (train only), numpy + pandas (serve), Next.js 15 (App Router), Recharts, Vercel.

## Global Constraints

- **Free tier only.** No always-on backend, no managed database, no paid infra.
- **Frontend on Vercel; Next.js App Router. NO Gradio.**
- **No Supabase, Render, or Fly.io.**
- **Real market data only** in the product (`data/prices.parquet`). Synthetic series are allowed *inside unit tests only*.
- **Serve-time Python deps = numpy + pandas only.** torch, stable-baselines3, gymnasium, yfinance belong to `requirements-train.txt` and must never be imported by `api/` or `rl/backtest.py` / `rl/policy.py`.
- **Universe (fixed, 9 ETFs, order matters):** `SPY, QQQ, IWM, EFA, EEM, AGG, TLT, GLD, VNQ`.
- **Chronological split:** train `2013-01-01 → 2020-12-31`; out-of-sample `2021-01-01 → latest`.
- **Lookback = 30 trading days. Long-only, fully invested, weights sum to 1.**
- **Profiles:** `conservative`, `balanced`, `aggressive`.

---

## File Structure

```
rl/
  __init__.py
  metrics.py        # pure metric functions (Sharpe, max drawdown, CAGR, ...)
  backtest.py       # run_backtest + baselines (numpy/pandas, no torch)
  policy.py         # NumpyMLPPolicy: npz -> weights (no torch)
  env.py            # PortfolioEnv (Gymnasium) — training only
  data_pull.py      # one-time real-data fetch -> data/prices.parquet
  train.py          # trains 3 PPO agents, exports models/<profile>.npz
  tests/
    __init__.py
    test_metrics.py
    test_backtest.py
    test_env.py
    test_policy.py
    test_export_parity.py
data/
  prices.parquet    # real market snapshot (committed)
  UNIVERSE.py       # single source of truth for ticker list + split dates
models/
  conservative.npz  balanced.npz  aggressive.npz
api/
  backtest.py       # Vercel Python function (numpy + backtest, no torch)
app/                # Next.js App Router (repo root)
  page.tsx
  layout.tsx
  components/BacktestDashboard.tsx
  components/__tests__/BacktestDashboard.test.tsx
package.json
requirements.txt        # serve-time: numpy, pandas
requirements-train.txt  # training: torch, stable-baselines3, gymnasium, yfinance, pandas, pyarrow
vercel.json             # bundles rl/, data/, models/ into the python function
pytest.ini
.gitignore
README.md
```

**Deviation from spec §10:** Next.js lives at the repo root (not `web/`) because a Vercel Python function must sit at the deployment root under `/api`, and a single-root project is the least-config way to make Next.js + a Python function coexist.

---

### Task 1: Scaffold repo and shared constants

**Files:**
- Create: `.gitignore`, `pytest.ini`, `requirements.txt`, `requirements-train.txt`
- Create: `rl/__init__.py`, `rl/tests/__init__.py`, `data/UNIVERSE.py`

**Interfaces:**
- Produces: `data/UNIVERSE.py` exposing `TICKERS: list[str]`, `PROFILES: list[str]`, `TRAIN_END: str`, `OOS_START: str`, `LOOKBACK: int`, `DATA_START: str`.

- [ ] **Step 1: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
venv/
node_modules/
.next/
.vercel/
*.zip
.DS_Store
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = rl/tests
python_files = test_*.py
```

- [ ] **Step 3: Create `requirements.txt` (serve-time only)**

```
numpy>=1.26
pandas>=2.1
pyarrow>=15.0
```

- [ ] **Step 4: Create `requirements-train.txt`**

```
-r requirements.txt
torch>=2.2
stable-baselines3>=2.3
gymnasium>=0.29
yfinance>=0.2.40
pandas-datareader>=0.10
```

- [ ] **Step 5: Create `data/UNIVERSE.py`**

```python
"""Single source of truth for the asset universe and train/test split."""

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT", "GLD", "VNQ"]
PROFILES = ["conservative", "balanced", "aggressive"]

DATA_START = "2013-01-01"
TRAIN_END = "2020-12-31"
OOS_START = "2021-01-01"

LOOKBACK = 30
N_ASSETS = len(TICKERS)
```

- [ ] **Step 6: Create empty `rl/__init__.py` and `rl/tests/__init__.py`**

Both files empty.

- [ ] **Step 7: Verify pytest collects (0 tests) and constants import**

Run: `python -c "from data.UNIVERSE import TICKERS; assert len(TICKERS)==9; print('ok')" && pytest -q`
Expected: prints `ok`; pytest reports `no tests ran`.

- [ ] **Step 8: Commit**

```bash
git add .gitignore pytest.ini requirements.txt requirements-train.txt rl/__init__.py rl/tests/__init__.py data/UNIVERSE.py
git commit -m "chore: scaffold repo, deps, and universe constants"
```

---

### Task 2: Metric functions

**Files:**
- Create: `rl/metrics.py`
- Test: `rl/tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `total_return(equity: np.ndarray) -> float`
  - `cagr(equity: np.ndarray, periods_per_year: int = 252) -> float`
  - `annual_vol(returns: np.ndarray, periods_per_year: int = 252) -> float`
  - `sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float` (risk-free = 0)
  - `max_drawdown(equity: np.ndarray) -> float` (returned as a negative fraction, e.g. -0.25)
  - `equity` is a 1-D array of cumulative portfolio value normalized to start at 1.0; `returns` is a 1-D array of per-period **simple** returns.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rl/tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rl/tests/test_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add rl/metrics.py rl/tests/test_metrics.py
git commit -m "feat: add financial metric functions with edge-case guards"
```

---

### Task 3: Backtest engine and baselines

**Files:**
- Create: `rl/backtest.py`
- Test: `rl/tests/test_backtest.py`

**Interfaces:**
- Consumes: `rl.metrics.*`.
- Produces:
  - `Policy = Callable[[np.ndarray], np.ndarray]` — takes an observation vector, returns a length-`N_ASSETS` weight vector (already softmaxed, sums to 1).
  - `build_observation(returns_window: np.ndarray, current_weights: np.ndarray) -> np.ndarray` — flattens `(LOOKBACK, N) + (N,)` into a `(LOOKBACK*N + N,)` vector.
  - `run_backtest(prices: pd.DataFrame, policy: Policy, *, lookback: int, rebalance: str, cost: float = 0.001) -> dict` where `prices` is a DateTimeIndexed frame of adjusted closes (columns in `TICKERS` order), already sliced to the desired window. Returns:
    ```python
    {
      "dates": list[str],            # ISO dates aligned to equity
      "equity": list[float],         # normalized to 1.0 at start
      "weights_timeline": list[{"date": str, "weights": list[float]}],
      "metrics": {"total_return","cagr","annual_vol","sharpe","max_drawdown","avg_turnover"},
    }
    ```
  - `equal_weight_backtest(prices, *, rebalance="M", cost=0.001) -> dict` — same shape, ignores policy.
  - `buy_and_hold(prices, ticker: str) -> dict` — same shape, single asset held.
  - `rebalance` accepts `"D"`, `"W"`, `"M"` (pandas offset aliases for daily/weekly/monthly).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rl/tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.backtest'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
    log_returns = np.log1p(simple.values)  # (T, N)
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

    equity = np.array(equity[1:]) if len(equity) > 1 else np.array([1.0])
    equity = equity / equity[0]
    ret = np.array(port_returns)
    out_dates = [d.date().isoformat() for d in dates[start:]]
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
```

Note: `log_returns` is computed but only `simple` returns are used for portfolio math; drop the `log_returns` line if the linter flags it (harmless either way).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rl/tests/test_backtest.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add rl/backtest.py rl/tests/test_backtest.py
git commit -m "feat: add numpy backtest engine with equal-weight and buy-and-hold baselines"
```

---

### Task 4: PortfolioEnv (Gymnasium, training only)

**Files:**
- Create: `rl/env.py`
- Test: `rl/tests/test_env.py`

**Interfaces:**
- Consumes: `rl.backtest.build_observation`, `data.UNIVERSE` constants.
- Produces: `PortfolioEnv(gym.Env)` constructed as
  `PortfolioEnv(simple_returns: np.ndarray, *, lookback=30, episode_len=252, lambda_risk: float, lambda_cost: float, cost: float = 0.001, seed=None)`.
  - `observation_space`: `Box(low=-inf, high=inf, shape=(lookback*N + N,))`.
  - `action_space`: `Box(low=-10, high=10, shape=(N,))` (wide bounds so PPO's mean is never clipped — matters for export parity in Task 7).
  - Static method `weights_from_action(action) -> np.ndarray` = softmax.
  - `reset()` / `step()` follow Gymnasium 5-tuple API.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rl/tests/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.env'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
        self._recent = None

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
        self._recent = []
        return self._obs(), {}

    def step(self, action):
        new_w = self.weights_from_action(action)
        turnover = float(np.abs(new_w - self._weights).sum())
        port_ret = float((self.returns[self._t] * new_w).sum())
        self._recent.append(port_ret)
        if len(self._recent) > self.lookback:
            self._recent.pop(0)
        variance = float(np.var(self._recent)) if len(self._recent) > 1 else 0.0

        reward = np.log1p(port_ret) - self.lambda_risk * variance - self.lambda_cost * self.cost * turnover

        self._weights = new_w
        self._t += 1
        terminated = False
        truncated = (self._t - self._start) >= self.episode_len or self._t >= self.T - 1
        return self._obs(), float(reward), terminated, truncated, {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rl/tests/test_env.py -v`
Expected: 4 passed. (Requires `pip install -r requirements-train.txt`.)

- [ ] **Step 5: Commit**

```bash
git add rl/env.py rl/tests/test_env.py
git commit -m "feat: add PortfolioEnv with reward shaping knobs"
```

---

### Task 5: Numpy policy inference

**Files:**
- Create: `rl/policy.py`
- Test: `rl/tests/test_policy.py`

**Interfaces:**
- Consumes: nothing (numpy only).
- Produces: `NumpyMLPPolicy` with:
  - `NumpyMLPPolicy(layers: list[tuple[np.ndarray, np.ndarray]])` where each tuple is `(W, b)`, `W` shape `(out, in)`. Hidden layers use `tanh`; the final layer is linear; output is softmaxed to weights.
  - classmethod `from_npz(path) -> NumpyMLPPolicy` reading keys `w0,b0,w1,b1,...,wK,bK` (K = last index).
  - `__call__(self, obs: np.ndarray) -> np.ndarray` returning softmaxed weights summing to 1.
  - `raw_action(self, obs) -> np.ndarray` returning the pre-softmax output (needed for Task 7 parity test).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from rl.policy import NumpyMLPPolicy


def test_forward_softmax_sums_to_one():
    w0 = np.eye(3); b0 = np.zeros(3)
    w1 = np.eye(3); b1 = np.zeros(3)
    pol = NumpyMLPPolicy([(w0, b0), (w1, b1)])
    out = pol(np.array([1.0, 2.0, 3.0]))
    assert abs(out.sum() - 1.0) < 1e-9
    assert (out >= 0).all()


def test_raw_action_matches_manual_two_layer():
    w0 = np.array([[1.0, 0.0], [0.0, 1.0]]); b0 = np.array([0.1, -0.1])
    w1 = np.array([[2.0, 0.0], [0.0, 2.0]]); b1 = np.array([0.0, 0.0])
    pol = NumpyMLPPolicy([(w0, b0), (w1, b1)])
    x = np.array([0.5, -0.5])
    h = np.tanh(w0 @ x + b0)     # hidden uses tanh
    expected = w1 @ h + b1       # last layer linear
    assert np.allclose(pol.raw_action(x), expected)


def test_from_npz_roundtrip(tmp_path):
    p = tmp_path / "m.npz"
    np.savez(p, w0=np.eye(2), b0=np.zeros(2), w1=np.eye(2), b1=np.zeros(2))
    pol = NumpyMLPPolicy.from_npz(str(p))
    out = pol(np.array([1.0, 2.0]))
    assert out.shape == (2,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rl/tests/test_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.policy'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Serve-time policy inference. numpy only (no torch)."""
import numpy as np


def _softmax(a):
    a = np.asarray(a, float)
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


class NumpyMLPPolicy:
    def __init__(self, layers):
        self.layers = [(np.asarray(w, float), np.asarray(b, float)) for w, b in layers]

    @classmethod
    def from_npz(cls, path):
        data = np.load(path)
        layers = []
        i = 0
        while f"w{i}" in data:
            layers.append((data[f"w{i}"], data[f"b{i}"]))
            i += 1
        return cls(layers)

    def raw_action(self, obs):
        x = np.asarray(obs, float)
        for k, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if k < len(self.layers) - 1:
                x = np.tanh(x)
        return x

    def __call__(self, obs):
        return _softmax(self.raw_action(obs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rl/tests/test_policy.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add rl/policy.py rl/tests/test_policy.py
git commit -m "feat: add numpy MLP policy inference for serve time"
```

---

### Task 6: Real-data pull

**Files:**
- Create: `rl/data_pull.py`
- Create (output, committed): `data/prices.parquet`

**Interfaces:**
- Consumes: `data.UNIVERSE`.
- Produces: `fetch_prices(tickers, start, end=None) -> pd.DataFrame` (adjusted close, columns in `tickers` order, business-day index, forward-filled gaps, no NaN); `main()` writes `data/prices.parquet`.

- [ ] **Step 1: Write `rl/data_pull.py`**

```python
"""One-time fetch of REAL adjusted-close prices -> data/prices.parquet.
Run manually: python -m rl.data_pull
Training deps only (yfinance). Not imported at serve time."""
import sys
import pandas as pd
from data.UNIVERSE import TICKERS, DATA_START


def fetch_prices(tickers, start, end=None) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close[tickers]  # enforce column order
    close = close.asfreq("B").ffill().dropna()
    if close.isna().any().any():
        raise ValueError("NaNs remain after ffill/dropna")
    return close


def main():
    prices = fetch_prices(TICKERS, DATA_START)
    if len(prices) < 1500:
        print(f"WARNING: only {len(prices)} rows fetched", file=sys.stderr)
    prices.to_parquet("data/prices.parquet")
    print(f"Wrote data/prices.parquet: {prices.shape}, "
          f"{prices.index.min().date()} -> {prices.index.max().date()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the fetch**

Run: `pip install -r requirements-train.txt && python -m rl.data_pull`
Expected: prints a shape like `(3000+, 9)` and a date range starting `2013-01`. If yfinance fails, fall back: implement the `import yfinance` block with a `pandas_datareader.stooq` alternative and rerun.

- [ ] **Step 3: Sanity-check the committed data**

Run:
```bash
python -c "import pandas as pd; d=pd.read_parquet('data/prices.parquet'); print(d.shape); print(list(d.columns)); assert list(d.columns)==['SPY','QQQ','IWM','EFA','EEM','AGG','TLT','GLD','VNQ']; assert not d.isna().any().any(); print('ok')"
```
Expected: prints shape, column list, `ok`.

- [ ] **Step 4: Commit (code + the real dataset)**

```bash
git add rl/data_pull.py data/prices.parquet
git commit -m "feat: add real-data fetch and commit ETF price snapshot"
```

---

### Task 7: Train agents and export weights (with parity gate)

**Files:**
- Create: `rl/train.py`
- Create (output, committed): `models/conservative.npz`, `models/balanced.npz`, `models/aggressive.npz`
- Test: `rl/tests/test_export_parity.py`

**Interfaces:**
- Consumes: `rl.env.PortfolioEnv`, `rl.policy.NumpyMLPPolicy`, `data.UNIVERSE`.
- Produces:
  - `PROFILE_PARAMS: dict[str, dict]` mapping each profile to `{"lambda_risk","lambda_cost"}`.
  - `export_policy(model) -> list[tuple[np.ndarray, np.ndarray]]` — extracts `mlp_extractor.policy_net` Linear layers + `action_net` from an SB3 PPO model, in numpy, matching `NumpyMLPPolicy` layer order (hidden tanh, final linear).
  - `train_profile(profile, train_returns, timesteps) -> PPO` and `main()`.

- [ ] **Step 1: Write the failing parity test**

```python
import numpy as np
from rl.env import PortfolioEnv
from rl.policy import NumpyMLPPolicy
from rl.train import export_policy


def test_numpy_export_matches_sb3(tmp_path):
    from stable_baselines3 import PPO
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0003, 0.01, size=(400, 9)).astype(np.float32)
    env = PortfolioEnv(returns, lookback=30, episode_len=50, lambda_risk=0.1, lambda_cost=0.1)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, n_epochs=1, seed=0,
                policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=128)

    layers = export_policy(model)
    npz = tmp_path / "m.npz"
    np.savez(npz, **{f"w{i}": w for i, (w, b) in enumerate(layers)},
             **{f"b{i}": b for i, (w, b) in enumerate(layers)})
    pol = NumpyMLPPolicy.from_npz(str(npz))

    obs, _ = env.reset(seed=1)
    sb3_action, _ = model.predict(obs, deterministic=True)
    assert np.allclose(pol.raw_action(obs), sb3_action, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rl/tests/test_export_parity.py -v`
Expected: FAIL with `ImportError`/`ModuleNotFoundError` on `rl.train.export_policy`.

- [ ] **Step 3: Write `rl/train.py`**

```python
"""Trains 3 reward-shaped PPO agents and exports numpy policies.
Run: python -m rl.train   (training deps required)."""
import numpy as np
import pandas as pd
from data.UNIVERSE import TICKERS, TRAIN_END, DATA_START, LOOKBACK

PROFILE_PARAMS = {
    "conservative": {"lambda_risk": 8.0, "lambda_cost": 4.0},
    "balanced":     {"lambda_risk": 2.0, "lambda_cost": 1.0},
    "aggressive":   {"lambda_risk": 0.1, "lambda_cost": 0.1},
}


def _train_returns():
    prices = pd.read_parquet("data/prices.parquet")
    prices = prices.loc[DATA_START:TRAIN_END, TICKERS]
    return prices.pct_change().fillna(0.0).values.astype(np.float32)


def export_policy(model):
    """Extract policy_net (tanh MLP) + action_net (linear mean) as numpy layers."""
    net = model.policy.mlp_extractor.policy_net
    layers = []
    for module in net:                      # Sequential of Linear/Tanh
        if module.__class__.__name__ == "Linear":
            w = module.weight.detach().cpu().numpy()
            b = module.bias.detach().cpu().numpy()
            layers.append((w, b))
    action = model.policy.action_net         # final Linear -> action mean
    layers.append((action.weight.detach().cpu().numpy(),
                   action.bias.detach().cpu().numpy()))
    return layers


def train_profile(profile, train_returns, timesteps=300_000):
    from stable_baselines3 import PPO
    from rl.env import PortfolioEnv
    params = PROFILE_PARAMS[profile]
    env = PortfolioEnv(train_returns, lookback=LOOKBACK, episode_len=252,
                       lambda_risk=params["lambda_risk"], lambda_cost=params["lambda_cost"])
    model = PPO("MlpPolicy", env, seed=0, verbose=0,
                policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=timesteps)
    return model


def main():
    import os
    os.makedirs("models", exist_ok=True)
    train_returns = _train_returns()
    for profile in PROFILE_PARAMS:
        model = train_profile(profile, train_returns)
        layers = export_policy(model)
        path = f"models/{profile}.npz"
        np.savez(path,
                 **{f"w{i}": w for i, (w, b) in enumerate(layers)},
                 **{f"b{i}": b for i, (w, b) in enumerate(layers)})
        print(f"exported {path} ({len(layers)} layers)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run parity test to verify it passes**

Run: `pytest rl/tests/test_export_parity.py -v`
Expected: 1 passed. If it fails on `net_arch`/`action_net` naming, print `model.policy` and adjust `export_policy` to the actual module names — the parity assertion is the gate.

- [ ] **Step 5: Train for real and commit models**

Run: `python -m rl.train`
Expected: three `exported models/<profile>.npz` lines.

```bash
git add rl/train.py rl/tests/test_export_parity.py models/conservative.npz models/balanced.npz models/aggressive.npz
git commit -m "feat: train 3 reward-shaped PPO agents and export numpy policies"
```

---

### Task 8: Vercel Python API function

**Files:**
- Create: `api/backtest.py`
- Test: `rl/tests/test_api.py`

**Interfaces:**
- Consumes: `rl.backtest`, `rl.policy`, `data.UNIVERSE`.
- Produces:
  - `handle(body: dict) -> tuple[int, dict]` — pure function: validates `{profile,start,end,rebalance}`, runs the backtest, returns `(status_code, json_dict)`. Testable without HTTP.
  - `class handler(BaseHTTPRequestHandler)` — Vercel entry point that reads the POST body, calls `handle`, writes JSON.
  - Success payload: `{"agent": <backtest dict>, "baselines": {"equal_weight": <dict>, "spy": <dict>}}`. Error payload: `{"error": "..."}` with status 400.

- [ ] **Step 1: Write the failing test**

```python
import json
from api.backtest import handle


def test_bad_profile_returns_400():
    status, body = handle({"profile": "nope", "start": "2021-01-01",
                           "end": "2021-06-01", "rebalance": "M"})
    assert status == 400
    assert "error" in body


def test_bad_dates_return_400():
    status, body = handle({"profile": "balanced", "start": "2021-06-01",
                           "end": "2021-01-01", "rebalance": "M"})
    assert status == 400


def test_valid_request_returns_series_and_baselines():
    status, body = handle({"profile": "balanced", "start": "2021-01-01",
                           "end": "2021-12-31", "rebalance": "M"})
    assert status == 200
    assert len(body["agent"]["equity"]) > 100
    assert set(body["baselines"]) == {"equal_weight", "spy"}
    for row in body["agent"]["weights_timeline"]:
        assert abs(sum(row["weights"]) - 1.0) < 1e-6


def test_determinism():
    req = {"profile": "aggressive", "start": "2021-01-01", "end": "2021-12-31", "rebalance": "W"}
    a = handle(req); b = handle(req)
    assert json.dumps(a[1]) == json.dumps(b[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rl/tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.backtest'`.
(If `api` is not importable, add an empty `api/__init__.py` — but Vercel treats `api/*.py` as functions; keep `__init__.py` out of the deploy via `.vercelignore` if it causes routing issues. For tests, run from repo root so `api.backtest` resolves.)

- [ ] **Step 3: Write `api/backtest.py`**

```python
"""Vercel Python serverless function: POST /api/backtest.
Serve-time deps: numpy, pandas only."""
import json
import os
from http.server import BaseHTTPRequestHandler

import pandas as pd

from rl.backtest import run_backtest, equal_weight_backtest, buy_and_hold
from rl.policy import NumpyMLPPolicy
from data.UNIVERSE import TICKERS, PROFILES, LOOKBACK, OOS_START

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PRICES = None
_POLICIES = {}


def _prices():
    global _PRICES
    if _PRICES is None:
        _PRICES = pd.read_parquet(os.path.join(_ROOT, "data", "prices.parquet"))[TICKERS]
    return _PRICES


def _policy(profile):
    if profile not in _POLICIES:
        _POLICIES[profile] = NumpyMLPPolicy.from_npz(
            os.path.join(_ROOT, "models", f"{profile}.npz"))
    return _POLICIES[profile]


def handle(body: dict):
    profile = body.get("profile")
    rebalance = body.get("rebalance", "M")
    start = body.get("start", OOS_START)
    end = body.get("end")

    if profile not in PROFILES:
        return 400, {"error": f"profile must be one of {PROFILES}"}
    if rebalance not in ("D", "W", "M"):
        return 400, {"error": "rebalance must be D, W, or M"}
    try:
        window = _prices().loc[start:end]
    except Exception:
        return 400, {"error": "invalid date range"}
    if len(window) <= LOOKBACK + 5:
        return 400, {"error": "date range too short or out of bounds"}

    agent = run_backtest(window, _policy(profile), lookback=LOOKBACK, rebalance=rebalance)
    baselines = {
        "equal_weight": equal_weight_backtest(window, rebalance="M"),
        "spy": buy_and_hold(window, "SPY"),
    }
    return 200, {"agent": agent, "baselines": baselines}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            status, payload = 400, {"error": "invalid JSON"}
        else:
            status, payload = handle(body)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rl/tests/test_api.py -v`
Expected: 4 passed.

- [ ] **Step 5: Full test suite green**

Run: `pytest -q`
Expected: all tasks' tests pass.

- [ ] **Step 6: Commit**

```bash
git add api/backtest.py rl/tests/test_api.py
git commit -m "feat: add Vercel python backtest function with validation"
```

---

### Task 9: Next.js frontend

**Files:**
- Create: `package.json`, `next.config.mjs`, `tsconfig.json`, `app/layout.tsx`, `app/page.tsx`
- Create: `app/components/BacktestDashboard.tsx`
- Test: `app/components/__tests__/BacktestDashboard.test.tsx`, `vitest.config.ts`

**Interfaces:**
- Consumes: `POST /api/backtest` returning `{agent, baselines}` (Task 8 contract).
- Produces: a client dashboard component that renders controls, calls the API, and draws the equity curve + weights area chart + metric cards.

> Design note for the implementer: load the frontend-design skill before styling, and the dataviz skill before writing the charts. Follow their guidance for palette, typography, and chart form.

- [ ] **Step 1: Create `package.json`**

```json
{
  "name": "rl-portfolio-optimizer",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "test": "vitest run"
  },
  "dependencies": {
    "next": "^15.0.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "recharts": "^2.12.0"
  },
  "devDependencies": {
    "@testing-library/react": "^16.0.0",
    "@testing-library/jest-dom": "^6.4.0",
    "@types/react": "^18.3.0",
    "jsdom": "^24.0.0",
    "typescript": "^5.4.0",
    "vitest": "^2.0.0"
  }
}
```

- [ ] **Step 2: Create `tsconfig.json`, `next.config.mjs`, `vitest.config.ts`**

`next.config.mjs`:
```js
/** @type {import('next').NextConfig} */
export default { reactStrictMode: true };
```

`tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020", "lib": ["dom", "dom.iterable", "ES2020"],
    "jsx": "preserve", "module": "esnext", "moduleResolution": "bundler",
    "strict": true, "esModuleInterop": true, "skipLibCheck": true,
    "noEmit": true, "plugins": [{ "name": "next" }]
  },
  "include": ["**/*.ts", "**/*.tsx"], "exclude": ["node_modules"]
}
```

`vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
export default defineConfig({ test: { environment: "jsdom", globals: true } });
```

- [ ] **Step 3: Write the failing component test**

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import BacktestDashboard from "../BacktestDashboard";

const fakeResponse = {
  agent: { dates: ["2021-01-04"], equity: [1.0],
    weights_timeline: [{ date: "2021-01-04", weights: Array(9).fill(1/9) }],
    metrics: { total_return: 0.1, cagr: 0.1, annual_vol: 0.12, sharpe: 0.8, max_drawdown: -0.15, avg_turnover: 0.2 } },
  baselines: {
    equal_weight: { dates: ["2021-01-04"], equity: [1.0], weights_timeline: [], metrics: { total_return: 0.05, cagr: 0.05, annual_vol: 0.1, sharpe: 0.5, max_drawdown: -0.2, avg_turnover: 0.1 } },
    spy: { dates: ["2021-01-04"], equity: [1.0], weights_timeline: [], metrics: { total_return: 0.07, cagr: 0.07, annual_vol: 0.15, sharpe: 0.6, max_drawdown: -0.25, avg_turnover: 0.0 } },
  },
};

test("renders metrics after a run", async () => {
  global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve(fakeResponse) })) as any;
  render(<BacktestDashboard />);
  screen.getByRole("button", { name: /run backtest/i }).click();
  await waitFor(() => expect(screen.getByText(/sharpe/i)).toBeInTheDocument());
});
```

- [ ] **Step 4: Run test to verify it fails**

Run: `npm install && npm test`
Expected: FAIL — cannot find `../BacktestDashboard`.

- [ ] **Step 5: Write `app/components/BacktestDashboard.tsx`**

```tsx
"use client";
import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area } from "recharts";

const PROFILES = ["conservative", "balanced", "aggressive"] as const;
const TICKERS = ["SPY","QQQ","IWM","EFA","EEM","AGG","TLT","GLD","VNQ"];

type Series = { dates: string[]; equity: number[];
  weights_timeline: { date: string; weights: number[] }[];
  metrics: Record<string, number> };
type Resp = { agent: Series; baselines: { equal_weight: Series; spy: Series } };

export default function BacktestDashboard() {
  const [profile, setProfile] = useState<typeof PROFILES[number]>("balanced");
  const [start, setStart] = useState("2021-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [rebalance, setRebalance] = useState("M");
  const [data, setData] = useState<Resp | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setLoading(true); setError(null);
    try {
      const res = await fetch("/api/backtest", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile, start, end, rebalance }) });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "request failed");
      setData(json);
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  }

  const equityData = data?.agent.dates.map((d, i) => ({
    date: d, agent: data.agent.equity[i],
    equalWeight: data.baselines.equal_weight.equity[i],
    spy: data.baselines.spy.equity[i] })) ?? [];

  const weightData = data?.agent.weights_timeline.map((row) => {
    const o: any = { date: row.date };
    TICKERS.forEach((t, i) => (o[t] = row.weights[i]));
    return o; }) ?? [];

  return (
    <div style={{ maxWidth: 960, margin: "0 auto", padding: 24 }}>
      <h1>RL Portfolio Optimizer</h1>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
        {PROFILES.map((p) => (
          <button key={p} onClick={() => setProfile(p)}
            aria-pressed={profile === p}
            style={{ fontWeight: profile === p ? 700 : 400 }}>{p}</button>
        ))}
        <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
        <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
        <select value={rebalance} onChange={(e) => setRebalance(e.target.value)}>
          <option value="D">Daily</option><option value="W">Weekly</option><option value="M">Monthly</option>
        </select>
        <button onClick={run} disabled={loading}>{loading ? "Running…" : "Run backtest"}</button>
      </div>
      {error && <p role="alert" style={{ color: "crimson" }}>{error}</p>}
      {data && (
        <>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", margin: "16px 0" }}>
            {Object.entries(data.agent.metrics).map(([k, v]) => (
              <div key={k} style={{ border: "1px solid #ccc", padding: "8px 12px", borderRadius: 8 }}>
                <div style={{ fontSize: 12, textTransform: "capitalize" }}>{k.replace("_", " ")}</div>
                <div style={{ fontSize: 20, fontWeight: 700 }}>
                  {k.includes("return") || k.includes("drawdown") || k.includes("cagr") || k.includes("vol")
                    ? (v * 100).toFixed(1) + "%" : v.toFixed(2)}</div>
              </div>
            ))}
          </div>
          <h3>Equity curve (out-of-sample)</h3>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={equityData}>
              <XAxis dataKey="date" minTickGap={40} /><YAxis /><Tooltip /><Legend />
              <Line dataKey="agent" dot={false} /><Line dataKey="equalWeight" dot={false} />
              <Line dataKey="spy" dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <h3>Allocation over time</h3>
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={weightData}>
              <XAxis dataKey="date" minTickGap={40} /><YAxis /><Tooltip /><Legend />
              {TICKERS.map((t) => <Area key={t} dataKey={t} stackId="1" />)}
            </AreaChart>
          </ResponsiveContainer>
        </>
      )}
      <p style={{ fontSize: 12, color: "#666", marginTop: 24 }}>
        Backtest on held-out data (2021+). PPO agents trained on 2013–2020. Not investment advice.
      </p>
    </div>
  );
}
```

- [ ] **Step 6: Write `app/layout.tsx` and `app/page.tsx`**

`app/layout.tsx`:
```tsx
export const metadata = { title: "RL Portfolio Optimizer" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="en"><body>{children}</body></html>);
}
```

`app/page.tsx`:
```tsx
import BacktestDashboard from "./components/BacktestDashboard";
export default function Page() { return <BacktestDashboard />; }
```

- [ ] **Step 7: Run test to verify it passes**

Run: `npm test`
Expected: 1 passed.

- [ ] **Step 8: Commit**

```bash
git add package.json next.config.mjs tsconfig.json vitest.config.ts app/
git commit -m "feat: add Next.js dashboard with equity and allocation charts"
```

---

### Task 10: Vercel config, README, deploy

**Files:**
- Create: `vercel.json`, `.vercelignore`, `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a deployable Vercel project where `/api/backtest` bundles `rl/`, `data/`, `models/`.

- [ ] **Step 1: Create `vercel.json`**

```json
{
  "functions": {
    "api/backtest.py": {
      "runtime": "python3.12",
      "includeFiles": "{rl/**,data/**,models/**}"
    }
  }
}
```

- [ ] **Step 2: Create `.vercelignore`**

```
docs/
rl/tests/
requirements-train.txt
**/__pycache__/
```

- [ ] **Step 3: Write `README.md`**

Include: one-paragraph pitch; the reward-shaping table; **exact reproduction commands** (`pip install -r requirements-train.txt`, `python -m rl.data_pull`, `python -m rl.train`, `pytest`, `npm install && npm run dev`); an honest results section (fill in real OOS metrics per profile after Task 7); the out-of-sample/leakage note; "not investment advice." Every metric quoted must come from committed code + data.

- [ ] **Step 4: Local end-to-end check**

Run: `npm install && npm run dev`, open the app, run a backtest for each profile.
Expected: three profiles produce visibly different allocation charts; no console errors; API returns in a few seconds.

- [ ] **Step 5: Deploy to Vercel**

Run: `npx vercel --prod` (or connect the GitHub repo in the Vercel dashboard).
Expected: build succeeds; `/api/backtest` responds; the deployed dashboard runs a backtest.
If the function exceeds the size limit, confirm `.vercelignore` excludes `requirements-train.txt` and that only numpy/pandas are in `requirements.txt`.

- [ ] **Step 6: Commit**

```bash
git add vercel.json .vercelignore README.md
git commit -m "chore: add Vercel config, ignore rules, and README"
```

---

## Self-Review Notes

- **Spec coverage:** data snapshot (T6), universe/split (T1 constants), env+reward shaping (T4), 3 profiles (T7 `PROFILE_PARAMS`), numpy serving trick (T5) + parity gate (T7), backtest+baselines+metrics (T2/T3), Vercel Python API (T8), Next.js charts+explainer (T9), tests every task, deploy config (T10). All spec §3–§13 items map to a task.
- **Serve-time purity:** `api/backtest.py`, `rl/backtest.py`, `rl/policy.py`, `rl/metrics.py` import only numpy/pandas. torch/SB3/gymnasium appear only in `rl/env.py` and `rl/train.py` (training) — never bundled (excluded via `requirements.txt` split + `.vercelignore`).
- **Type consistency:** `Policy` = `obs -> weights`; `run_backtest` wraps a bare `policy(obs)` while `_walk` uses `weight_fn(window, current)` internally — the wrapper bridges them. `NumpyMLPPolicy.__call__` returns softmaxed weights (matches `Policy`); `raw_action` returns pre-softmax (matches SB3 mean for the parity test). npz keys `w{i}/b{i}` written in T7, read in T5 — consistent.
- **Known risk flagged in-plan:** SB3 module names in `export_policy` (T7 Step 4) — the parity test is the gate that catches any mismatch.
