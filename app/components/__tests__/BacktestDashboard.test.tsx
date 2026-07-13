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
