# TODO — RL Portfolio Optimizer

_Snapshot: 2026-07-12. Build is complete, tested, and deployed live._

## ✅ Done
- Full build: custom Gymnasium env, 3 reward-shaped PPO agents (real 2013–2020 ETF data), numpy-only serving, Vercel Python `/api/backtest`, Next.js dashboard.
- Reward function debugged (ex-ante Markowitz vol) → coherent conservative→aggressive risk spectrum.
- 29 Python + 1 frontend tests passing; final whole-branch review clean.
- Merged to `main`, feature branch deleted.
- Deployed to production and **verified live**: https://rl-portfolio-optimization-shiv-a.vercel.app (homepage 200, all 3 profiles return real backtests, validation 400s).
- Vercel Deployment Protection disabled → publicly accessible.

## 🔲 Remaining (actionable now)
1. **Push code to GitHub.** Local `main` (`72c0921`) is NOT on the remote — `origin/main` is gone. Recruiters clicking the repo would see stale/empty. Run: `git push -u origin main`.
2. **Confirm the GitHub repo is public** (Settings → General → Visibility) so recruiters can view the code.
3. **README** — generate a recruiter-facing README with the live link (in progress via readme skill).
4. **Interview-prep writeup** — `WHAT AND WHY/RL-Portfolio-Optimization/` (in progress).
5. **Add a screenshot/GIF** of the live dashboard to the README — a visual sells it far better than prose.

## 🌱 Optional / future (nice-to-have, not blocking)
- **Genuinely competitive RL** (the deferred deeper option): more timesteps, richer features (momentum/vol regime), reward/hyperparameter tuning to actually challenge the passive baselines. RL-for-trading is hard; no guarantee — but it's the honest stretch goal.
- **MLflow / experiment tracking** for the training runs (a clean resume bullet; deliberately skipped as YAGNI for the demo).
- **Custom domain** on Vercel for a cleaner link.
- **Cosmetic cleanups** (all flagged Minor, non-blocking): README defensive/equity footnote covers only 6 of 9 tickers (EFA/EEM/VNQ uncategorized); `rl/data_pull.py` broad `except`; `PortfolioEnv.reset()` deterministic-start edge for pathological configs.
- **Rebalance-frequency UX note:** the equal-weight baseline is fixed monthly regardless of the chosen cadence (intentional reference; document it if it confuses).
