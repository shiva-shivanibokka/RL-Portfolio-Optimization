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
