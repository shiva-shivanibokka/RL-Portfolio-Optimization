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
