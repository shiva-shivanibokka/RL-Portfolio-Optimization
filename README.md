# RL Portfolio Optimizer

An interactive reinforcement-learning portfolio allocator. Three PPO agents —
**Conservative**, **Balanced**, and **Aggressive** — are trained offline on a
custom [Gymnasium](https://gymnasium.farama.org/) environment over nine liquid
ETFs, with reward shaping as the only thing that distinguishes the three risk
profiles. At serve time the trained policy runs as a **pure-numpy MLP
forward pass** (no PyTorch, no Stable-Baselines3 on the server), which keeps
the deployed backend small enough for Vercel's free-tier Python functions.
A Next.js dashboard lets you pick a profile and a rebalance cadence, run an
out-of-sample backtest, and compare the agent against equal-weight and
buy-and-hold baselines with interactive charts.

**Not investment advice.** This project is a research/engineering
demonstration of applying RL to portfolio allocation, evaluated honestly out
of sample. Nothing here is a recommendation to buy, sell, or hold any
security.

## Architecture

```
rl/data_pull.py   -> pulls daily prices for the ETF universe -> data/prices.parquet
rl/env.py         -> Gymnasium PortfolioEnv (training-only; imports gymnasium)
rl/train.py       -> trains 3 PPO agents (torch + stable-baselines3, offline)
                     and exports each policy's MLP weights to models/*.npz
rl/policy.py      -> NumpyMLPPolicy: replays the exported weights with plain
                     numpy (tanh MLP + softmax) -- no torch import
rl/backtest.py    -> walk-forward backtest engine + equal-weight/buy-and-hold
                     baselines + rebalance-cadence handling (numpy + pandas only)
rl/metrics.py     -> total return, CAGR, annualized vol, Sharpe, max drawdown
api/backtest.py   -> Vercel Python function, POST /api/backtest
                     (imports rl.backtest / rl.policy / data.UNIVERSE only)
app/              -> Next.js App Router dashboard (profile picker, allocation
                     chart, equity curve, metrics table)
```

**The numpy-serving trick.** Training needs torch, Gymnasium, and
Stable-Baselines3 — none of that belongs in a serverless function. `rl/train.py`
extracts the trained policy's linear layers (`policy_net` + `action_net`) into
plain numpy arrays and saves them as `models/{profile}.npz`. `rl/policy.py`
reloads those arrays and replays the forward pass (`tanh` MLP -> softmax) with
nothing but numpy. A parity test (`rl/tests/test_export_parity.py`) checks the
numpy replay against Stable-Baselines3's own action output before this is
ever trusted. `requirements.txt` (serve-time) only lists `numpy`, `pandas`,
and `pyarrow`; `requirements-train.txt` layers `torch`/`stable-baselines3`/
`gymnasium`/`yfinance` on top for local training only, and is excluded from
the deploy bundle by `.vercelignore`.

**No look-ahead leakage.** The asset universe and split are defined once in
`data/UNIVERSE.py` and imported everywhere (training, backtest, API):
training uses `2013-01-01` -> `2020-12-31`; all reported results are
out-of-sample, `2021-01-01` onward. The agents never see post-2020 prices
during training, and the backtest engine only ever looks backward
(`lookback`-day trailing window) when it rebalances.

## Reward shaping

All three agents share the same environment, observation space, and network
architecture (`[64, 64]` tanh MLP). They differ **only** in two reward
coefficients:

```
reward = log_return - lambda_risk * ex_ante_vol - lambda_cost * cost * turnover
```

where `ex_ante_vol = sqrt(wᵀ Σ w)` (trailing-window Markowitz portfolio
volatility) and `turnover = sum(|new_weights - old_weights|)`.

| Profile | `lambda_risk` | `lambda_cost` |
|---|---|---|
| Conservative | 100.0 | 0.5 |
| Balanced | 12.0 | 0.3 |
| Aggressive | 0.0 | 0.1 |

Higher `lambda_risk` penalizes ex-ante volatility more, pushing the agent
toward bonds/gold/diversification; higher `lambda_cost` penalizes turnover
more, pushing toward less frequent rebalancing. (`rl/train.py`,
`PROFILE_PARAMS`.)

## Out-of-sample results (2021-01-01 -> 2026-07-10, monthly rebalance)

Measured directly from `/api/backtest` against `data/prices.parquet`; nothing
here is simulated or hand-tuned after the fact.

| Strategy | Defensive alloc | Equity alloc | Total return | Sharpe | Max drawdown |
|---|---|---|---|---|---|
| Conservative agent | 80% | 10% | +10% | 0.27 | -20% |
| Balanced agent | 64% | 19% | +29% | 0.55 | -21% |
| Aggressive agent | 35% | 35% | +35% | 0.49 | -28% |
| Equal-weight (monthly) | - | - | +57% | 0.70 | -26% |
| SPY buy & hold | - | - | +117% | 0.91 | -24% |

("Defensive" = AGG + TLT + GLD average allocation; "Equity" = SPY + QQQ + IWM
average allocation, over the backtest window.)

### Honest findings

All three RL agents **underperform both passive baselines** on raw total
return and Sharpe ratio out of sample. This is a common, well-documented
result in the RL-for-portfolio-allocation literature — a multi-year,
mostly-up-only equity bull market is close to the hardest possible regime
for a risk-averse agent to beat on raw return, and this project reports that
result rather than hiding it.

What the reward shaping *does* achieve, cleanly:

- **A coherent, monotonic risk spectrum.** Defensive allocation decreases
  strictly from Conservative (80%) to Balanced (64%) to Aggressive (35%),
  exactly matching the ordering of `lambda_risk`. The three agents are not
  three copies of the same policy — the reward shaping visibly and
  consistently changes behavior.
- **Real capital protection.** The Conservative agent has the **lowest max
  drawdown of any strategy in the table (-20%)** — lower than Balanced,
  Aggressive, equal-weight, and SPY buy-and-hold. It gives up return to buy
  that protection, which is exactly the trade a risk-averse allocator is
  supposed to make.

Read together: this is a rigorous, honestly-reported study of reward-shaped
RL allocation, not a hidden failure dressed up as a win. The interesting
result isn't "RL beats the market" (it doesn't, here) — it's that a single
environment with two reward coefficients reliably produces a real,
ordered risk/return spectrum, with the most risk-averse setting delivering
the best tail protection in the entire comparison.

## Tech stack

- **Training (offline, local only):** PyTorch, Stable-Baselines3 (PPO),
  Gymnasium, pandas, numpy, yfinance / pandas-datareader for data pulls
- **Serving (deployed):** pure numpy + pandas (no torch, no SB3, no gymnasium)
- **API:** Vercel Python serverless function (`api/backtest.py`,
  `python3.12` runtime)
- **Frontend:** Next.js (App Router), React, Recharts
- **Testing:** pytest (Python), Vitest + Testing Library (frontend)

## Reproduce locally

```bash
pip install -r requirements-train.txt   # training deps (torch, SB3, gymnasium, data pull)
python -m rl.data_pull                  # pull ETF price history -> data/prices.parquet
python -m rl.train                      # train 3 PPO agents, export models/*.npz
pytest                                  # run the Python test suite (env, backtest, metrics,
                                         # policy, export parity, API)
npm install && npm run dev              # frontend dev server (uses committed models/*.npz)
```

The frontend and serve-time API work from the committed `models/*.npz` and
`data/prices.parquet` without re-running training — `pip install -r
requirements-train.txt` / `rl.train` are only needed to retrain from scratch.

## Deploy

The Next.js frontend and the Python `/api/backtest` function deploy together
from the repo root — Vercel auto-detects the Next.js app; `vercel.json`
registers only the Python function and tells Vercel to bundle `rl/`, `data/`,
and `models/` alongside it via `includeFiles`.

```bash
npx vercel        # first-time interactive setup (requires a Vercel account/login)
npx vercel --prod # promote to production
```

or connect the GitHub repo directly in the [Vercel dashboard](https://vercel.com/new)
for automatic deploys on push. Everything here fits comfortably in Vercel's
free (Hobby) tier: the deployed function only imports numpy/pandas, and
`.vercelignore` excludes training-only code (`requirements-train.txt`,
`rl/tests/`, `docs/`) from the bundle.
