# RL Portfolio Optimizer — Design Spec

**Date:** 2026-07-12
**Project:** Project 8 (RL Agent for Portfolio Optimization) from the portfolio plan
**Status:** Approved for planning

## 1. Goal

An interactive web app where a visitor picks a **strategy profile** (Conservative /
Balanced / Aggressive) and a **backtest date window**, runs it, and sees a trained
PPO agent's portfolio allocation over time and its performance versus baselines —
computed live on request.

Demonstrates: reinforcement learning, PPO, custom Gymnasium environments,
reward shaping, financial backtesting, and shipping an ML model as a real product.

## 2. Constraints (locked)

- **Free tier only.** Everything runs on Vercel free tier.
- **Frontend on Vercel.** Next.js (App Router). **No Gradio.**
- **No Supabase, Render, or Fly.io.** No always-on backend, no managed DB.
- **Real market data only.** No synthetic data.

## 3. Data

- **Source:** one-time offline pull from a real provider (yfinance primary; Stooq via
  `pandas-datareader` as fallback). Committed to the repo as `data/prices.parquet`
  so nothing is fetched at serve time.
- **Universe (9 liquid ETFs across asset classes):**
  SPY, QQQ, IWM (US equity), EFA, EEM (international), AGG, TLT (bonds),
  GLD (gold), VNQ (real estate).
- **Range:** ~2013-01-01 through the latest available close (~2025).
- **Split (chronological — prevents look-ahead leakage):**
  - Train: 2013-01-01 → 2020-12-31
  - Out-of-sample: 2021-01-01 → latest. **The demo defaults to this OOS window**, so
    what a recruiter sees is genuinely held-out data the agent never trained on.
- Prices stored as adjusted close; daily simple + log returns derived at load time.

## 4. RL environment (`rl/env.py`)

Custom Gymnasium env, `PortfolioEnv`.

- **Observation:** trailing 30 trading days of daily returns for the 9 assets
  (30 × 9) **plus** the current portfolio weights (9), flattened → length 279 vector.
- **Action:** continuous 9-vector; `softmax` → long-only weights that sum to 1.
  No shorting, no leverage, fully invested (keeps the problem honest and bounded).
- **Step:** apply weights, realize next-day portfolio log return, charge transaction
  cost proportional to turnover `Σ|w_t − w_{t-1}|`.
- **Reward:** `r = portfolio_log_return − λ_risk · variance_proxy − λ_cost · turnover`
  where `variance_proxy` is the rolling variance of recent portfolio returns.
- **Episode:** a randomly-started window within the training period; fixed length
  (e.g., 252 trading days). Reset picks a new random start.

### Reward shaping → 3 profiles

Same PPO, same network, **only λ changes** (exact values tuned during implementation):

| Profile      | λ_risk | λ_cost | Expected behavior                          |
|--------------|--------|--------|--------------------------------------------|
| Conservative | high   | high   | Tilts to AGG/TLT/GLD, low turnover         |
| Balanced     | medium | medium | Mixed equity/bond allocation               |
| Aggressive   | low    | low    | Tilts to SPY/QQQ/IWM, chases return        |

Transaction cost assumption: ~10 bps (0.001) per unit turnover.

## 5. Training (`rl/train.py`, offline only)

- Stable-Baselines3 **PPO**, small MLP policy (e.g., 2×64).
- A few hundred thousand steps per profile — minutes on CPU.
- Trains 3 agents (one per profile) on the **training split only**.
- **Exports each policy's MLP weights to `models/<profile>.npz`** (see §6).
- torch/SB3 are **training-time dependencies only** — never shipped to serving.

## 6. The serving trick (keeps it on free tier)

torch + SB3 exceed serverless function size limits. So serving reimplements
inference without them:

- After training, extract the policy MLP's weight matrices and biases → `.npz`.
- **`rl/policy.py`** implements a ~30-line pure-numpy forward pass
  (`obs → tanh(W1·x+b1) → … → logits → softmax → weights`), matching PPO's
  deterministic action.
- **Serve-time dependencies = numpy + pandas only.** Tiny bundle, fast cold start,
  fully deterministic.
- A test asserts numpy inference == torch/SB3 inference on fixed observations, so
  the export can't silently drift.

## 7. Backtest engine (`rl/backtest.py`, shared)

Pure numpy/pandas. Input: a policy (numpy), a date window, rebalance frequency.
Walks the window day by day (re-querying the policy at each rebalance date, holding
weights between rebalances), applying transaction costs. Outputs:

- **Equity curve** (cumulative value, normalized to 1.0 at window start)
- **Weights timeline** (weights at each rebalance date)
- **Metrics:** total return, CAGR, annualized volatility, Sharpe (rf = 0, stated),
  max drawdown, average turnover.

**Baselines** computed over the same window:
- Equal-weight portfolio, rebalanced monthly.
- SPY buy-and-hold.

Rebalance options exposed to the user: daily / weekly / monthly.

## 8. API (`api/backtest.py` — Vercel Python Function)

- `POST /api/backtest`, body `{ profile, start, end, rebalance }`.
- Validates inputs (profile in the 3; dates within snapshot range; start < end).
- Loads `data/prices.parquet` + `models/<profile>.npz`, runs the backtest engine,
  returns JSON: `{ equity_curve, weights_timeline, metrics, baselines }`.
- Deterministic: identical inputs → identical output.
- Errors return a structured `{ error }` with a 4xx for bad input, never a 500 stack.

## 9. Frontend (`web/`, Next.js App Router on Vercel)

- Single dashboard page.
- **Controls:** 3 profile buttons; date-range picker (defaults to the OOS window,
  bounded to the snapshot range); rebalance dropdown; Run button.
- **Charts (Recharts):** equity curve (agent vs both baselines); stacked-area of
  weights over time; metric cards (CAGR, vol, Sharpe, max DD, turnover).
- **Explainer copy:** what PPO is; how reward shaping makes the 3 profiles differ;
  an explicit out-of-sample honesty note; "not investment advice" disclaimer.
- Loading + error states wired to the API contract.

## 10. Repo structure

```
rl/
  env.py            # Gymnasium PortfolioEnv
  train.py          # trains 3 PPO agents, exports .npz  (offline)
  backtest.py       # shared numpy backtest engine
  policy.py         # numpy MLP forward pass (serve-time inference)
  data_pull.py      # one-time real-data fetch -> data/prices.parquet
  tests/            # pytest
data/
  prices.parquet    # real market snapshot (committed)
models/
  conservative.npz  balanced.npz  aggressive.npz
api/
  backtest.py       # Vercel Python function (numpy + backtest, no torch)
web/                # Next.js app (Vercel)
requirements.txt        # serve-time: numpy, pandas
requirements-train.txt  # training-time: torch, stable-baselines3, gymnasium, yfinance
```

## 11. Testing

- **Env:** reset/step shapes correct; softmax weights sum to 1 and are ≥ 0;
  turnover cost reduces reward as expected.
- **Backtest:** metrics correct on a small hand-checkable price series
  (known return/vol/drawdown).
- **Export parity:** numpy inference == torch/SB3 inference on fixed observations.
- **Determinism:** same request → same JSON.
- **Frontend:** one smoke test that a mocked API response renders the charts.

## 12. Deliberately out of scope (YAGNI — add only on request)

- MLflow / TensorBoard experiment tracking (training-time only, invisible in demo).
- Live user-supplied tickers (RL policy is only valid for its trained universe).
- Shorting / leverage / margin.
- Multiple RL algorithms (A2C, SAC, etc.) — PPO only.
- Auth, databases, user accounts.

## 13. Success criteria

- Three profiles produce visibly different allocations on the same OOS window.
- At least one profile beats equal-weight on risk-adjusted return (Sharpe) OOS
  (if not, that's an honest finding to report, not a bug to hide).
- Whole thing deploys to Vercel free tier and runs a backtest in a few seconds.
- Every metric claim in the README is reproducible from committed code + data.
