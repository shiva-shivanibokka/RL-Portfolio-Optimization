"use client";
import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area, CartesianGrid } from "recharts";

const PROFILES = ["conservative", "balanced", "aggressive"] as const;
const TICKERS = ["SPY","QQQ","IWM","EFA","EEM","AGG","TLT","GLD","VNQ"];

// dataviz skill: 8-slot categorical palette (references/palette.md), CVD-validated
// via scripts/validate_palette.js. TICKERS has 9 entries; the skill notes a 9th
// series should fold into "Other" rather than get a generated hue, but this is a
// small, fixed asset-class set (not open cardinality), so it was extended with one
// more hue (sienna) and re-validated as a 9-slot set (all checks pass).
// ponytail: hand-picked 9th color instead of a generated-hue algorithm; revisit if
// the ticker universe grows beyond a handful of fixed asset classes.
const TICKER_COLORS: Record<string, string> = {
  SPY: "#2a78d6", QQQ: "#1baf7a", IWM: "#eda100", EFA: "#008300", EEM: "#4a3aa7",
  AGG: "#e34948", TLT: "#e87ba4", GLD: "#eb6834", VNQ: "#a0522d",
};
// Equity curve reuses three of the same validated slots (blue/green/orange) --
// all three clear 3:1 contrast on the light chart surface without needing relief.
const SERIES_COLORS = { agent: "#2a78d6", equalWeight: "#008300", spy: "#eb6834" };

type Series = { dates: string[]; equity: number[];
  weights_timeline: { date: string; weights: number[] }[];
  metrics: Record<string, number> };
type Resp = { agent: Series; baselines: { equal_weight: Series; spy: Series } };

const PCT_KEYS = ["return", "drawdown", "cagr", "vol"];
function formatMetric(key: string, value: number): string {
  return PCT_KEYS.some((k) => key.includes(k)) ? (value * 100).toFixed(1) + "%" : value.toFixed(2);
}

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
    } catch (e: any) { setError(e.message ?? "request failed"); } finally { setLoading(false); }
  }

  // Backend guarantees all three series share one date axis and length, so a
  // positional join by index is safe here -- see api/backtest.py.
  const equityData = data?.agent.dates.map((d, i) => ({
    date: d, agent: data.agent.equity[i],
    equalWeight: data.baselines.equal_weight.equity[i],
    spy: data.baselines.spy.equity[i] })) ?? [];

  const weightData = data?.agent.weights_timeline.map((row) => {
    const o: Record<string, string | number> = { date: row.date };
    TICKERS.forEach((t, i) => (o[t] = row.weights[i]));
    return o; }) ?? [];

  return (
    <main className="dashboard">
      <header className="header">
        <h1>RL Portfolio Optimizer</h1>
        <p className="subtitle">
          PPO agent vs. equal-weight and SPY baselines, backtested on held-out data.
        </p>
      </header>

      <section className="controls" aria-label="Backtest parameters">
        <div className="profile-group" role="group" aria-label="Risk profile">
          {PROFILES.map((p) => (
            <button
              key={p}
              type="button"
              className="profile-btn"
              onClick={() => setProfile(p)}
              aria-pressed={profile === p}
            >
              {p}
            </button>
          ))}
        </div>
        <label className="field">
          <span>Start</span>
          <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
        </label>
        <label className="field">
          <span>End</span>
          <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
        </label>
        <label className="field">
          <span>Rebalance</span>
          <select value={rebalance} onChange={(e) => setRebalance(e.target.value)}>
            <option value="D">Daily</option>
            <option value="W">Weekly</option>
            <option value="M">Monthly</option>
          </select>
        </label>
        <button type="button" className="run-btn" onClick={run} disabled={loading}>
          {loading ? "Running…" : "Run backtest"}
        </button>
      </section>

      {error && <p role="alert" className="error">{error}</p>}

      {data && (
        <>
          <section className="metrics" aria-label="Agent performance metrics">
            {Object.entries(data.agent.metrics).map(([k, v]) => (
              <div key={k} className="metric-card">
                <div className="metric-label">{k.replace(/_/g, " ")}</div>
                <div className="metric-value">{formatMetric(k, v)}</div>
              </div>
            ))}
          </section>

          <section aria-label="Equity curve">
            <h2>Equity curve (out-of-sample)</h2>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={equityData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#e1e0d9" vertical={false} />
                <XAxis dataKey="date" minTickGap={40} stroke="#898781" fontSize={12} />
                <YAxis stroke="#898781" fontSize={12} />
                <Tooltip />
                <Legend />
                <Line name="Agent" dataKey="agent" stroke={SERIES_COLORS.agent} strokeWidth={2} dot={false} />
                <Line name="Equal weight" dataKey="equalWeight" stroke={SERIES_COLORS.equalWeight} strokeWidth={2} dot={false} />
                <Line name="SPY" dataKey="spy" stroke={SERIES_COLORS.spy} strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </section>

          <section aria-label="Allocation over time">
            <h2>Allocation over time</h2>
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={weightData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#e1e0d9" vertical={false} />
                <XAxis dataKey="date" minTickGap={40} stroke="#898781" fontSize={12} />
                <YAxis stroke="#898781" fontSize={12} />
                <Tooltip />
                <Legend />
                {TICKERS.map((t) => (
                  <Area key={t} name={t} dataKey={t} stackId="1"
                    stroke={TICKER_COLORS[t]} fill={TICKER_COLORS[t]} fillOpacity={0.85} />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </section>
        </>
      )}

      <p className="disclaimer">
        Backtest on held-out data (2021+). PPO agents trained on 2013–2020 and never see the
        out-of-sample period during training. Past performance does not predict future results.
        Not investment advice.
      </p>

      <style dangerouslySetInnerHTML={{ __html: `
        .dashboard {
          max-width: 960px;
          margin: 0 auto;
          padding: 32px 24px 48px;
          font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          color: #0b0b0b;
        }
        .header h1 {
          font-size: 28px;
          font-weight: 700;
          margin: 0 0 4px;
        }
        .subtitle {
          color: #52514e;
          margin: 0 0 24px;
          font-size: 14px;
        }
        .controls {
          display: flex;
          align-items: flex-end;
          gap: 16px;
          flex-wrap: wrap;
          padding: 16px;
          background: #fcfcfb;
          border: 1px solid rgba(11, 11, 11, 0.1);
          border-radius: 10px;
          margin-bottom: 24px;
        }
        .profile-group {
          display: flex;
          gap: 8px;
        }
        .profile-btn {
          padding: 8px 14px;
          border-radius: 8px;
          border: 1px solid rgba(11, 11, 11, 0.15);
          background: #fff;
          font-size: 14px;
          text-transform: capitalize;
          cursor: pointer;
        }
        .profile-btn[aria-pressed="true"] {
          background: #2a78d6;
          color: #fff;
          border-color: #2a78d6;
          font-weight: 600;
        }
        .field {
          display: flex;
          flex-direction: column;
          gap: 4px;
          font-size: 12px;
          color: #52514e;
        }
        .field input,
        .field select {
          padding: 6px 8px;
          border-radius: 6px;
          border: 1px solid rgba(11, 11, 11, 0.15);
          font-size: 14px;
        }
        .run-btn {
          padding: 9px 18px;
          border-radius: 8px;
          border: none;
          background: #0b0b0b;
          color: #fff;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          margin-left: auto;
        }
        .run-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
        .error {
          color: #d03b3b;
          font-weight: 600;
          margin: 0 0 16px;
        }
        .metrics {
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
          margin: 8px 0 32px;
        }
        .metric-card {
          border: 1px solid rgba(11, 11, 11, 0.1);
          background: #fcfcfb;
          padding: 10px 16px;
          border-radius: 10px;
          min-width: 110px;
        }
        .metric-label {
          font-size: 12px;
          color: #898781;
          text-transform: capitalize;
        }
        .metric-value {
          font-size: 20px;
          font-weight: 700;
          margin-top: 2px;
        }
        h2 {
          font-size: 16px;
          font-weight: 600;
          margin: 24px 0 8px;
        }
        .disclaimer {
          font-size: 12px;
          color: #898781;
          margin-top: 32px;
          line-height: 1.5;
        }
      ` }} />
    </main>
  );
}
