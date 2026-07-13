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
    if not isinstance(body, dict):
        return 400, {"error": "request body must be a JSON object"}
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
    # NOTE (deviation from brief): baselines must be computed on the same
    # trailing window the agent actually trades, or their date axis and
    # length silently diverge from the agent's (agent's lookback=LOOKBACK
    # burns the first LOOKBACK-1 rows; baselines use lookback=1 internally).
    # Slicing off that same warmup here gives all three series an identical
    # date axis, each normalized to 1.0 at the same first date.
    base_window = window.iloc[LOOKBACK - 1:]
    baselines = {
        "equal_weight": equal_weight_backtest(base_window, rebalance="M"),
        "spy": buy_and_hold(base_window, "SPY"),
    }
    if not (agent["dates"] == baselines["equal_weight"]["dates"] == baselines["spy"]["dates"]):
        return 500, {"error": "internal alignment error"}
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
